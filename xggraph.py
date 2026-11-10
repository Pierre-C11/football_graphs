from datetime import datetime
from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    HPacker,
    TextArea,
)
from matplotlib.ticker import MultipleLocator
import store

HALF_TIME = 45 * 60
FULL_TIME = 90 * 60
TICK_INTERVAL = 5 * 60
OUTPUT_DIR = Path("data/graphs")
SHOW_INCIDENTS = True #Incidents feature on top of graph. Only displays reds now, see is_displayed_incidents and add_incident_markers

# Loading full content of 4 jsons needed for graph from eventid folder. todo: could be moved into parser?
def load_match_data(event_id):
    shotmap_data = store.load_part(event_id, "shotmap")
    incidents_data = store.load_part(event_id, "incidents")
    event_data = store.load_part(event_id, "event")
    lineup_data = store.load_part(event_id, "lineups")
    for name, data in [
        ("shotmap", shotmap_data),
        ("incidents", incidents_data),
        ("event", event_data),
        ("lineups", lineup_data),
    ]:
        if data is None:
            raise FileNotFoundError(
                f"Could not load {name}.json for event {event_id}"
            )
    event = event_data["event"]
    lineups = lineup_data
    shots = pd.json_normalize(shotmap_data["shotmap"])
    shots["xg"] = pd.to_numeric(
        shots["xg"], errors="coerce"
    ).fillna(0)
    shots["timeSeconds"] = pd.to_numeric(
        shots["timeSeconds"], errors="coerce"
    )
    # Stable sorting preserves the original order of shots sharing exactly the same timestamp.
    shots = (
        shots.dropna(subset=["timeSeconds"])
        .sort_values("timeSeconds", kind="stable")
        .reset_index(drop=True)
    )
    # Calculate cumulative xG independently for each team, every shot contributes separately including shots at identical timestamps.
    shots["cumulative_xg"] = (
        shots.groupby("isHome", sort=False)["xg"].cumsum()
    )
    incidents = incidents_data.get("incidents", [])

    return shots, incidents, event, lineups
# Parses metadata from event.json and lineups.json
def get_metadata(event):
    return {
        "home": event["homeTeam"]["name"],
        "away": event["awayTeam"]["name"],
        "home_score": event["homeScore"]["current"],
        "away_score": event["awayScore"]["current"],
        "date": datetime.fromtimestamp(
            event["startTimestamp"]
        ).strftime("%Y-%m-%d"),
        "competition": event["tournament"]["uniqueTournament"]["name"],
    }

def get_team_colours(lineups):
    colours = {}
    for side in ("home", "away"):
        team_colours = lineups[f"{side}"]["playerColor"]
        colours[side] = {
            "primary": "#" + team_colours["primary"],
            "secondary": "#" + team_colours["fancyNumber"],
        }
    return colours

# Incident labels helper methods. Checks whether benchTime exists, if it does it is used instead of time (needed for card incidents to staff)
def incident_time(incident):
    if incident.get("benchTime") is not None:
        return incident["benchTime"]
    return incident.get("time")

# Adds added time to incident time if there is any. todo: check how this is used and how incidents records time for added time events
def incident_seconds(incident):
    
    time = incident_time(incident)
    if time is None:
        return None
    added_time = incident.get("addedTime") or 0
    return (time + added_time) * 60

# checks whether incident was first half or 2nd half: returns 0 (1st Half), 1 (2nd Half)
def incident_half(incident):
    time = incident_time(incident)
    if time is None:
        return None
    return 0 if time <= 45 else 1

# Checks what name is available in case shortName is not
def player_name(player):
    if not player:
        return "Unknown"
    return (
        player.get("shortName")
        or player.get("name")
        or "Unknown"
    )

# returns length of injury time in minutes depending on which half is current. Only implemented for regular time as of now
def get_injury_time(incidents, half):
    if half ==0:
        scheduled_minute = 45
    elif half == 1:
        scheduled_minute = 90
    else: scheduled_minute = 90     #todo: Change this to support extra time periods
    for incident in incidents:
        if (
            incident.get("incidentType") == "injuryTime"
            and incident.get("time") == scheduled_minute
        ):
            return incident.get("length", 0) or 0
    return 0

# Used to set x_axis endpoints. Gathers list of incidents and their times to find halves endpoints
def get_timed_incidents(incidents, half):
    """
    Returns timed relevant incidents. Goals and fouls still affect graph boundary, even though not displayed in the incident labels
    """
    incident_types = {
        "goal",
        "substitution",
        "card",
        "foul",
    }
    result = []
    for incident in incidents:
        if incident.get("incidentType") not in incident_types:
            continue
        if incident_half(incident) != half:
            continue
        seconds = incident_seconds(incident)
        if seconds is not None:
            result.append((seconds, incident))
    return result

# Returns the latest incident of any type in each half and latest shot in each half
def get_half_endpoint(incidents, half, shots):
    """
    The graph boundary is the latest of:
    - scheduled half end plus recorded injury time;
    - latest relevant incident;
    - latest shot plotted in that half.
    """
    scheduled_end = HALF_TIME if half == 0 else FULL_TIME
    injury_time = get_injury_time(incidents, half)
    injury_endpoint = scheduled_end + injury_time * 60
    timed = get_timed_incidents(incidents, half)
    latest_incident = max(
        (seconds for seconds, _ in timed),
        default=scheduled_end,
    )
    if half == 0:
        half_shots = shots[shots["timeSeconds"] <= HALF_TIME]
    else:
        half_shots = shots[shots["timeSeconds"] > HALF_TIME]
    latest_shot = (
        half_shots["timeSeconds"].max()
        if not half_shots.empty
        else scheduled_end
    )
    return max(injury_endpoint, latest_incident, latest_shot)

# Gathers list of all assists. necessary because goals info is within shotmap, assists within incidents
def get_goal_assists(incidents):
    assists = {}
    for incident in incidents:
        if incident.get("incidentType") != "goal":
            continue
        assister = incident.get("assist1")
        if not assister:
            continue
        scorer = incident.get("player") or {}
        seconds = incident.get("reversedPeriodTime")  #uses reversedPeriodTime as key: to go between lineups and incidents
        if seconds is None:
            continue
        key = (int(seconds), scorer.get("shortName"))
        assists[key] = assister
    return assists

def format_goal_time(seconds):
    total_seconds = int(seconds)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"

def goal_label(row, assists):
    scorer = row.get("player.shortName", "Unknown")
    if pd.isna(scorer):
        scorer = "Unknown"

    goal_time = format_goal_time(row["timeSeconds"]) + " "

    goal_type = str(row.get("goalType", "")).lower()
    is_own_goal = "own" in goal_type

    if is_own_goal:
        goal_time = f"OG {goal_time}"

    assist_key = (
        int(row["reversedPeriodTime"]), # uses reversedPeriodTime as key: to go between lineups and incidents
        row.get("player.shortName"),
    )
    assister = assists.get(assist_key)
    if assister:
        assist_text = f"\nassist: {assister.get("shortName")}"
    else:
        assist_text = ""

    label = f"{goal_time}{scorer}{assist_text}"
    return label

def get_xg_at_time(shots, time):
    if shots.empty:
        return 0.0
    idx = shots["timeSeconds"].searchsorted(time, side="right") - 1
    if idx < 0:
        return 0.0
    return shots["cumulative_xg"].iloc[idx]

def add_goal_annotations(ax, team_shots, shots, team_colour, assists, ymax):
    texts = []
    goals = team_shots[     #goals looked at for labels here are only own team's
        team_shots["goalType"].notna()
        & (team_shots["goalType"] != "")
    ]
    #which team's shots we processing? Need to isolate all other shots from this team's shots, for label moving
    is_home = team_shots["isHome"].iloc[0] if not team_shots.empty else None
    oppo_shots = shots[shots["isHome"] != is_home]

    #setting limits for label mover alg.
    x_min, x_max = ax.get_xlim()
    max_x_shift = 900  #max total distance shift allowed in x (for y = ymax) in seconds superfluous with step*5 < 900
    x_shift_step = 120
    y_shift_step = 0.05
    min_y_distance = 0.2 #TOO CLOSE value. min distance tolerable between label's y and 2 curves. Loop does not worry about xdistance

    for _, shot in goals.iterrows():
        x = shot["timeSeconds"]
        y = shot["cumulative_xg"]
        if not oppo_shots.empty:
            oppo_y = get_xg_at_time(oppo_shots, x)
        else:
            oppo_y = 0

        label = goal_label(shot, assists)   #construct the label text       
        #shift_direction: 3 cases depending where is space: above lines, between or below? Here: basic first shift leftup, left, or rightdown
        if y >= oppo_y or (y < abs(oppo_y - y) and y < 0.8*ymax):
            shift_direction_x = -1  #left
            shift_direction_y = 1   #up
            label_x = x - 700
            label_y = y + ymax * 0.05
        elif y < abs(oppo_y - y):
            shift_direction_x = 1   #right
            shift_direction_y = 0   #stall
            label_x = x - 150
            label_y = y + ymax * 0.04 #slightly up: can only get here if y is smaller than oppo_y
        else:
            shift_direction_x = 1  #right
            shift_direction_y = -1  #down
            label_x = x + 600
            label_y = y * 0.5 + (-0.2 * abs(oppo_y - y) / ymax)

        counter = 0 #loop counter to maintain max 5
        x_shifted = 0   #shift distance tracker
        y_shifted = 0

        while (
            abs(label_y - get_xg_at_time(team_shots, label_x)) < min_y_distance
            or abs(label_y - get_xg_at_time(oppo_shots, label_x)) < min_y_distance
        ):
            if counter > 5: break   #max 5 runs of loop
            counter+=1
            if abs(label_y - y) > 0.4: break #max label xG change tolerated since start of loop
            if x_shifted + x_shift_step > max_x_shift:
                break
            
            new_x = label_x + shift_direction_x * x_shift_step  #moving
            new_y = label_y + shift_direction_y * y_shift_step  #shift_direction_y set to 0 absorbs this if middle case

            if not x_min + 240 <= new_x <= x_max - 240: #240: horizontal margin. cannot go closer than 240 seconds to edge of fig
                break
            if not 0 <= new_y <= ymax:   #vertical boundaries
                break

            label_x = new_x
            label_y = new_y
            x_shifted += x_shift_step
            y_shifted += y_shift_step

        scorer_text = ax.annotate(
            label,
            xy=(x, y),
            xytext=(label_x, label_y),
            ha="center",
            va="bottom",
            fontsize=10.5,
            linespacing=1.15,
            color=team_colour["secondary"],
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": team_colour["primary"],
                "edgecolor": team_colour["secondary"],
                "linewidth": 1.4,
                "alpha": 0.4,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": team_colour["secondary"],
                "linewidth": 0.8,
                "shrinkA": 1,
                "shrinkB": 0,
            },
            zorder=1,
        )
        texts.append(scorer_text)

    return texts

# Axis configuration
def format_clock(seconds):
    minutes = int(seconds // 60)
    return f"{minutes}'"

def configure_half_axis(ax, half, endpoint, incidents):
    start = 0 if half == 0 else HALF_TIME
    scheduled_end = HALF_TIME if half == 0 else FULL_TIME
    injury_time = get_injury_time(incidents, half)
    injury_endpoint = scheduled_end + injury_time * 60
    ax.set_xlim(start, endpoint)
    first_tick = int(np.ceil(start / TICK_INTERVAL) * TICK_INTERVAL)
    ticks = list(
        np.arange(
            first_tick,
            scheduled_end,
            TICK_INTERVAL,
            dtype=int,
        )
    )
    # Only add the regular-time label separately when injury time exists.
    if injury_time:
        ticks.append(scheduled_end)
    # Preserve the existing final injury-time endpoint label.
    ticks.append(injury_endpoint)
    ticks = sorted(set(ticks))
    labels = []
    for tick in ticks:
        if tick == injury_endpoint and injury_time:
            previousTick = labels.pop()
            previousTick = previousTick[:-1]
            labels.append(previousTick)
            labels.append(
                f"      +{injury_time}'"
            )
        else:
            labels.append(format_clock(tick))
    ax.tick_params(axis="x", labelsize=9.5)   #change size of minutes tick labels
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    # Grid lines follow both x-axis and y-axis ticks.
    ax.grid(axis="x", alpha=0.25)

#Decide which incidents receive a visible label. EDIT THIS FUNCTION to change the selected incident types/classes.
def is_displayed_incident(incident):
    incident_type = incident.get("incidentType")
    incident_class = incident.get("incidentClass")
    if incident_type == "substitution":             
        return False        #for now, subs not included
    if incident_type == "card":
        if incident_class == "yellow":
            return False    #for now, yellows not included
        else:
            return True
    if incident_type == "goal" and incident_class == "penalty":
        return False        #for now, penalties not included
    return False

# vertical incident lines for reds (eventually yellows, subs, pens?) can handle yellows but they're turned off in is_displayed
def add_incident_markers(ax, incidents, half):
    selected = []
    #isolating which incidents are selected
    for i in incidents:
        if not is_displayed_incident(i):
            continue
        if incident_half(i) != half:
            continue
        seconds = incident_seconds(i)
        if seconds is None:
            continue
        else:
            selected.append(i)

    #todo: check if previous incident in selected was recent, if so, merge them/change label height/something for legibility
    for incident in selected:
        incident_type = incident.get("incidentType")
        incident_class = incident.get("incidentClass")
        description = ""

        if incident_type == "goal":
            description = "Penalty"
            color = "black"
        if incident_type == "card":
            color = "goldenrod"
            if incident_class in {"red", "yellowRed"}:
                color = "red"    
            if incident.get("benchTime") is not None:
                description = f"{incident.get("manager").get("shortName")}"
            else:
                description = f"{incident.get("player").get("shortName")}"

        ax.axvline(
            seconds,          #add + 20 to move text 20 seconds back/forth
            ymax=1.0,           #default value 1.0. change to extend lines above ymax.
            color = color,
            linewidth=0.8,
            alpha=0.75,
            linestyle=(0, (2, 2)),
            clip_on=False,
        )
        ax.text(
            seconds,
            1.0175,
            description,
            color = color,
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=10.5,
        )

# xG lines and goal markers
def add_team_plot(ax, shots, half, is_home, colour, endpoint):
    #Plots cumulative xG. Every shot contributes an individual increase to cumulative xG. Only goals receive visible markers.
    half_start = 0 if half == 0 else HALF_TIME

    team_shots = shots[
        (shots["isHome"] == is_home)
        & (shots["timeSeconds"] <= endpoint)
    ].copy()
    previous_shots = team_shots[
        team_shots["timeSeconds"] < half_start
    ]
    starting_xg = previous_shots["xg"].sum()
    half_shots = team_shots[
        (team_shots["timeSeconds"] >= half_start)
        & (team_shots["timeSeconds"] <= endpoint)
    ].copy()

    if half_shots.empty:
        return half_shots
    
    # Construct the staircase explicitly.
    x_values = [half_start]
    y_values = [starting_xg]
    previous_xg = starting_xg
    for _, shot in half_shots.iterrows():
        shot_time = shot["timeSeconds"]
        shot_cumulative_xg = shot["cumulative_xg"]
        x_values.extend([
            shot_time,
            shot_time,
        ])
        y_values.extend([
            previous_xg,
            shot_cumulative_xg,
        ])
        previous_xg = shot_cumulative_xg

    # Only extend the first-half staircase to its graph boundary.
    if half == 0:
        x_values.append(endpoint)
        y_values.append(previous_xg)

    # Draw the cumulative xG staircase.
    ax.plot(
        x_values,
        y_values,
        color=colour["primary"],
        linewidth=6,
        path_effects=[
            pe.Stroke(
                linewidth=8.5,
                foreground=colour["secondary"],
            ),
            pe.Normal(),
        ],
        zorder=3,
    )
    
    # Draw markers ONLY for goals.
    goals = half_shots[
        half_shots["goalType"].notna()
        & (half_shots["goalType"] != "")
    ]

    if not goals.empty:
        ax.scatter(
            goals["timeSeconds"].to_numpy(),
            goals["cumulative_xg"].to_numpy(),
            s=78,
            facecolor=colour["primary"],
            edgecolor=colour["secondary"],
            linewidth=2,
            zorder=5,
        )

    return half_shots

# Final xG labels outside the right side of the graph. todo: move these and make better
def add_total_xg_label(ax, shots, is_home, colour, endpoint, metadata, ymax):
    total_xg = shots.loc[
        (shots["isHome"] == is_home)
        & (shots["timeSeconds"] <= endpoint),
        "xg",
    ].sum()

    if is_home:
        team_name = metadata["home"].rsplit(" ", 1)[0]
    else:
        team_name = metadata["away"].rsplit(" ", 1)[0]
    
    label_transform = ax.transAxes
    y_position = total_xg/ymax  #setting label height as final xG total
    ax.text(
        0.995,
        y_position,
        f"{team_name} {total_xg:.2f}xG",
        transform=label_transform,
        ha="left",
        va="top",
        fontsize=12.5,
        color=colour["secondary"],
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": colour["primary"],
            "edgecolor": colour["secondary"],
            "linewidth": 1.0,
            "alpha": 0.95,
        },
    )

# Match title. todo: center around whitespace
def add_match_title(fig, metadata, colours):
    def title_text(text, team_colour):
        return TextArea(
            text,
            textprops={
                "color": team_colour["primary"],
                "fontsize": 32.5,
                "path_effects": [
                    pe.Stroke(
                        linewidth=3,
                        foreground=team_colour["secondary"],
                    ),
                    pe.Normal(),
                ],
                "family": "Cambria",
            },
        )

    title = HPacker(
        children=[
            title_text(f"{metadata["home"]} {metadata['home_score']}", colours["home"]), 
            title_text(f" {metadata['away_score']} {metadata["away"]}", colours["away"]),
        ],
        align="center",
        pad=0,
        sep=1,
    )
    title_anchor = AnchoredOffsetbox(
        loc="upper center",
        child=title,
        frameon=False,
        bbox_to_anchor=(0.485, 0.975),
        bbox_transform=fig.transFigure,
        borderpad=0,
    )
    fig.add_artist(title_anchor)

    #Left subtitle (competition & league info)
    fig.text(
        0.99,
        0.01,
        f"{metadata['competition']} {metadata['date']}\nFrom @pchevalier.bsky.social",
        ha="right",
        va="bottom",
        color="black",
        fontsize=9.5,
        style= "italic",
    )
    
# Main function
def generate_worm_graph(event_id):
    shots, incidents, event, lineups = load_match_data(event_id)
    metadata = get_metadata(event)
    colours = get_team_colours(lineups)
    assists = get_goal_assists(incidents)

    endpoints = {
        0: get_half_endpoint(incidents, 0, shots),
        1: get_half_endpoint(incidents, 1, shots),
    }

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10, 8),                #controls printed graph size
        sharey=True,
        gridspec_kw={
            "width_ratios": [1, 1],
            "wspace": 0.21,
        },
    )

    fig.patch.set_facecolor("white")
    graph_background = "#d9d9d9"
    
    for ax in (ax1, ax2):
        ax.set_facecolor(graph_background)

    axes = {
        0: ax1,
        1: ax2,
    }
    max_xg = (
        shots["cumulative_xg"].max()
        if not shots.empty
        else 0
    )
    ymax = max(max_xg * 1.1, 0.5)

    for half, ax in axes.items():
        endpoint = endpoints[half]
        home_shots = add_team_plot(
            ax,
            shots,
            half,
            True,
            colours["home"],
            endpoint,
        )
        away_shots = add_team_plot(
            ax,
            shots,
            half,
            False,
            colours["away"],
            endpoint,
        )
        add_goal_annotations(
            ax,
            home_shots,
            shots,
            colours["home"],
            assists,
            ymax,
        )
        add_goal_annotations(
            ax,
            away_shots,
            shots,
            colours["away"],
            assists,
            ymax,
        )
        configure_half_axis(
            ax,
            half,
            endpoint,
            incidents,
        )
        if SHOW_INCIDENTS:    
            add_incident_markers(
                ax,
                incidents,
                half,
            )
        ax.yaxis.set_major_locator(MultipleLocator(0.2))    #y axis tick marks and grid setting
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(axis="y", which="major", alpha=0.25)
        ax.grid(axis="y", which="minor", alpha=0.25)
        ax.set_ylim(0, ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Final accumulated xG labels appear only to the right of the second-half graph.
    add_total_xg_label(
        ax2,
        shots,
        True,
        colours["home"],
        endpoints[1],
        metadata,
        ymax,
    )
    add_total_xg_label(
        ax2,
        shots,
        False,
        colours["away"],
        endpoints[1],
        metadata,
        ymax,
    )

    #tick and spine parameters
    ax1.set_ylabel("Accumulated xG", labelpad=12, fontsize=14)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", which="major", left=False, labelleft=False)
    ax2.tick_params(axis="y", which="minor", left=False, labelleft=False)
    ax1.tick_params(axis="y", which="major", left=True, labelleft=True)
    ax1.tick_params(axis="y", which="minor", left=True, labelleft=False)

    # Half-time and full-time markers.
    ax1.axvline(
        HALF_TIME,
        color="grey",
        linestyle="--",
        linewidth=1.85,
        alpha=0.5,
    )
    ax1.text(
        HALF_TIME,
        0.98,
        "HT",
        transform=ax1.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        alpha=0.4,
    )
    ax2.axvline(
        FULL_TIME,
        color="grey",
        linestyle="--",
        linewidth=1.85,
        alpha=0.5,
    )
    ax2.text(
        FULL_TIME,
        0.98,
        "FT",
        transform=ax2.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        alpha=0.4,
    )

    add_match_title(fig, metadata, colours)
    fig.text(                        #bottom axis title adjusted here
        endpoints[0] + 40,
        -0.075,
        "Match time (minutes)",
        transform=ax1.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=13.5
    )
    fig.subplots_adjust(            #margins around plot adjusted here. need to change to fit in substitutinos
        top=0.85,
        bottom=0.14,
        left=0.09,
        right=0.81,
        wspace=0.08,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{metadata['home']} vs {metadata['away']} "
        f"{metadata['date']}.png"
    )
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    output_path = OUTPUT_DIR / filename
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    
    plt.show()
    print(f"Saved graph to: {output_path}")

    return fig, (ax1, ax2)

# Command-line entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python worm_graph.py EVENT_ID"
        )
    generate_worm_graph(int(sys.argv[1]))
