from datetime import datetime
from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    HPacker,
    TextArea,
    VPacker,
)
import store

HALF_TIME = 45 * 60
FULL_TIME = 90 * 60
TICK_INTERVAL = 5 * 60
OUTPUT_DIR = Path("data/graphs")

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
# Returns string with player name and jersey number of players for subtitutes labels
def player_label(player):
    if not player:
        return "Unknown"
    name = player_name(player)
    jersey = player.get("jerseyNumber")
    if jersey is not None:
        return f"{name}, #{jersey}"
    return name

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

    jersey = row.get("player.jerseyNumber")
    goal_type = str(row.get("goalType", "")).lower()
    is_own_goal = "own" in goal_type

    if is_own_goal:
        scorer = f"Own goal: {scorer}"
    if pd.notna(jersey):
        scorer += f", #{jersey}"

    body_part = row.get("bodyPart")
    if pd.notna(body_part) and body_part != "other":
        details = str(body_part)
    else:
        details = ""
        
    goal_time = format_goal_time(row["timeSeconds"])
    if details:
        details += f", {goal_time}"
    else:
        details = goal_time

    assist_key = (
        int(row["reversedPeriodTime"]), # uses reversedPeriodTime as key: to go between lineups and incidents
        row.get("player.shortName"),
    )
    assister = assists.get(assist_key)
    if assister:
        assist_text = f"\nAssist: {player_label(assister)}"
    else:
        assist_text = ""

    label = f"{scorer}\n{details}{assist_text}"
    return label

def add_goal_annotations(ax, team_shots, oppo_shots, team_colour, assists, ymax):
    texts = []
    goals = team_shots[
        team_shots["goalType"].notna()
        & (team_shots["goalType"] != "")
    ]
    for _, shot in goals.iterrows():
        x = shot["timeSeconds"]
        y = shot["cumulative_xg"]

        #checking for closest shot in timeseconds from oppo_shots. initiating at 0
        if not oppo_shots.empty:
            oppo_idx = (oppo_shots["timeSeconds"] - x).abs().idxmin()
            oppo_y = oppo_shots.loc[oppo_idx, "cumulative_xg"]
        else:
            oppo_y = 0

        label = goal_label(shot, assists)

        #checks whether scoringTeam xG at time of goal is greater or smaller than otherTeam xG. if greater: draw label up, left diagonal
        #if space under scoringTeamxG is bigger than space between both xG totals, draw label down, right diagonal
        #if oppoxG greater than scoringTeamxG but space between both xG totals greater than space under scoringTeamxG: draw label left
        if (y - oppo_y) >= 0:
            label_x = x - 285
            label_y = y + ymax * 0.035
        elif y > (oppo_y - y):
            label_x = x + 250
            label_y = y * 0.75 - ymax * 0.02125
        else:
            label_x = x - 300
            label_y = y + ymax * 0.035

        scorer_text = ax.text(
            label_x,
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            linespacing=1.2,
            color=team_colour["secondary"],
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": team_colour["primary"],
                "edgecolor": team_colour["secondary"],
                "linewidth": 1.4,
                "alpha": 0.4,
            },
            zorder=6,
        )     
        texts.append(scorer_text)   #builds array of labels

    if texts:            #todo:check what these do. visually, arrows to goallabels seem disabled
        adjust_text(
            texts,
            ax=ax,
            arrowprops={
                "arrowstyle": "-",  
                "color": "grey",
                "lw": 0.7,
            },
        )

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
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    # Grid lines follow both x-axis and y-axis ticks.
    ax.grid(axis="both", alpha=0.25)

# xG lines and goal markers
def add_team_plot(ax, shots, half, is_home, colour, endpoint):
    """
    Plots cumulative xG. Every shot contributes an individual increase to cumulative xG. Only goals receive visible markers.
    Repeated timestamps are deliberately retained. If several shots share a timestamp, the staircase makes successive vertical jumps
    First half: Extend horizontal line to graph boundary. Second half: Stop at the team's final shot. todo: implement extra time halves
    """
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
        linewidth=2.2,
        path_effects=[
            pe.Stroke(
                linewidth=3.8,
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
            s=38,
            facecolor=colour["primary"],
            edgecolor=colour["secondary"],
            linewidth=1.2,
            zorder=5,
        )
    return half_shots

# Final xG labels outside the right side of the graph. todo: move these and make better
def add_total_xg_label(ax, shots, is_home, colour, endpoint):
    total_xg = shots.loc[
        (shots["isHome"] == is_home)
        & (shots["timeSeconds"] <= endpoint),
        "xg",
    ].sum()

    team_name = "Home" if is_home else "Away"
    label_transform = ax.transAxes
    if is_home: # Move the home label down by 5 display pixels. only use of ScaledTranslation. inches, so divide pixels by figure DPI.
        y_position = 1.0 - (5 * ax.figure.dpi / 72) / ax.bbox.height
    else:   # Raise the away label to bring it closer to the home label.
        y_position = 0.94

    ax.text(
        1.02,
        y_position,
        f"{team_name} xG: {total_xg:.2f}",
        transform=label_transform,
        ha="left",
        va="top",
        fontsize=9,
        color=colour["secondary"],
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": colour["primary"],
            "edgecolor": colour["secondary"],
            "linewidth": 1.3,
            "alpha": 0.95,
        },
    )

# Match title. todo: center around whitespace
def add_match_title(fig, metadata, colours, ax, endpoint):
    def title_text(text, team_colour):
        return TextArea(
            text,
            textprops={
                "color": team_colour["primary"],
                "fontsize": 14,
                "weight": "bold",
                "path_effects": [
                    pe.Stroke(
                        linewidth=2.5,
                        foreground=team_colour["secondary"],
                    ),
                    pe.Normal(),
                ],
            },
        )
    home_title = title_text(
        metadata["home"],
        colours["home"],
    )
    home_score = title_text(
        str(metadata["home_score"]),
        colours["home"],
    )
    separator = TextArea(
        " vs ",
        textprops={
            "color": "black",
            "fontsize": 15,
            "weight": "bold",
        },
    )
    away_score = title_text(
        str(metadata["away_score"]),
        colours["away"],
    )
    away_title = title_text(
        metadata["away"],
        colours["away"],
    )
    top_title = HPacker(
        children=[
            home_title,
            TextArea(" "),
            home_score,
            separator,
            away_score,
            TextArea(" "),
            away_title,
        ],
        align="center",
        pad=0,
        sep=1,
    )
    subtitle = HPacker(
            children=[
                TextArea(f"{metadata['competition']}", textprops = {"fontsize":11}),
                TextArea(f"  |   ", textprops = {"fontsize":11}),
                TextArea(f"{metadata['date']}", textprops = {"fontsize":11}),
            ],
            align="center",
            pad=0,
            sep=1,    
    )
    title = VPacker(
        children=[
            top_title,
            subtitle,
        ],
        align="center",
        pad=0,
        sep=3,
    )
    anchored_title = AnchoredOffsetbox(
        loc="lower center",
        child=title,
        frameon=False,
        bbox_to_anchor=(endpoint + 40, 1.0),
        bbox_transform=ax.get_xaxis_transform(),
        borderpad=0,
    )
    fig.add_artist(anchored_title)
    
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
        figsize=(10, 7),                #controls printed graph size
        sharey=True,
        gridspec_kw={
            "width_ratios": [1, 1],
            "wspace": 0.11,
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
    ymax = max(max_xg * 1.15, 0.5)

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
            home_shots, away_shots,
            colours["home"],
            assists,
            ymax,
        )
        add_goal_annotations(
            ax,
            away_shots, home_shots,
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
    )
    add_total_xg_label(
        ax2,
        shots,
        False,
        colours["away"],
        endpoints[1],
    )

    ax1.set_ylabel("Accumulated xG")
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", left=False, labelleft=False)

    # Half-time and full-time markers.
    ax1.axvline(
        HALF_TIME,
        color="grey",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )
    ax1.text(
        HALF_TIME,
        0.98,
        "HT",
        transform=ax1.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
    )
    ax2.axvline(
        FULL_TIME,
        color="grey",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )
    ax2.text(
        FULL_TIME,
        0.98,
        "FT",
        transform=ax2.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
    )

    add_match_title(fig, metadata, colours, ax1, endpoints[0])
    fig.text(                        #bottom axis title adjusted here
        endpoints[0] + 40,
        -0.08,
        "Match time (minutes)",
        transform=ax1.get_xaxis_transform(),
        ha="center",
        va="top",
    )
    fig.subplots_adjust(            #margins around plot adjusted here. need to change to fit in substitutinos
        top=0.9,
        bottom=0.1,
        left=0.07,
        right=0.82,
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
        dpi=200,
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
