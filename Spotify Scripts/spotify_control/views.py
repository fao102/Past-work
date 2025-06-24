# views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpRequest, JsonResponse
from datetime import datetime
import re
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from .forms import SpotifyPlaylistForm
from typing import Dict, List
from django.utils.timezone import make_aware


def create_spotify():
    auth_manager = SpotifyOAuth(
        client_id="94966b3f99094e03892a3bee1d35e3c4",
        client_secret="8379f65c85b44f3a894aab704d98f6f2",
        redirect_uri="http://127.0.0.1:8000/callback/",  # Adjust this URL as needed
        scope="playlist-modify-public",
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)

    return auth_manager, sp


def main(request: HttpRequest):
    context: Dict[str, any]
    context = {}
    form = SpotifyPlaylistForm()
    tracks_to_remove = []
    errors = []
    auth_manager, sp = create_spotify()

    if request.method == "POST":
        form = SpotifyPlaylistForm(request.POST)
        if form.is_valid():

            orig_playlist_link = form.cleaned_data["orig_playlist_link"]
            new_playlist_link = form.cleaned_data["new_playlist_link"]
            date = form.cleaned_data["date"].date()

            orig_playlist_id = get_playlist_id(orig_playlist_link)
            new_playlist_id = get_playlist_id(new_playlist_link)

            if orig_playlist_id and new_playlist_id:

                # Get all tracks from the original playlist
                all_tracks = get_all_tracks(orig_playlist_id, sp)

                # Filter tracks based on the provided date
                filtered_tracks = filter_tracks_by_date(all_tracks, date)
                filtered_track_uris = filter_tracks_uri_by_date(all_tracks, date)

                # Display the tracks that will be affected

                for track in filtered_tracks:
                    if (
                        track is not None
                        and isinstance(track, dict)
                        and "track" in track
                        and isinstance(track["track"], dict)
                        and "uri" in track["track"]
                    ):
                        tracks_to_remove.append(
                            (
                                track["track"]["name"],
                                "-",
                                track["track"]["artists"][0]["name"],
                            )
                        )

                add_tracks_to_playlist(new_playlist_id, filtered_track_uris, sp)

                # remove_tracks_from_playlist(orig_playlist_id, filtered_track_uris, sp)
        else:
            errors.append(form.errors)
            form = SpotifyPlaylistForm()

    context["errors"] = errors
    context["form"] = form
    context["tracks"] = tracks_to_remove
    return render(request, "main.html", context)


def get_playlist_id(playlist_link):
    match = re.search(r"playlist/([^/?]+)", playlist_link)
    if match:
        return match.group(1)
    else:
        return None


def get_all_tracks(playlist_id, spotify):
    tracks = []
    offset = 0
    while True:
        response = spotify.playlist_tracks(playlist_id, offset=offset)
        tracks += response["items"]
        if response["next"]:
            offset += len(response["items"])
        else:
            break
    return tracks


# Function to filter tracks added in 2020
def filter_tracks_uri_by_date(tracks, date):
    filtered_tracks = []
    for track in tracks:

        added_at = datetime.strptime(track["added_at"].split("T")[0], "%Y-%m-%d").date()

        if added_at < date:
            if (
                track is not None
                and isinstance(track, dict)
                and "track" in track
                and isinstance(track["track"], dict)
                and "uri" in track["track"]
            ):
                filtered_tracks.append(track["track"]["uri"])
    return filtered_tracks


def filter_tracks_by_date(tracks, date):
    filtered_tracks = []
    for track in tracks:
        added_at = datetime.strptime(track["added_at"].split("T")[0], "%Y-%m-%d").date()

        if added_at < date:
            filtered_tracks.append(track)
    return filtered_tracks


def add_tracks_to_playlist(playlist_id, tracks, sp):
    batch_size = 100  # Number of tracks to add per request (maximum is 100)
    for i in range(0, len(tracks), batch_size):
        batch = tracks[i : i + batch_size]
        sp.playlist_add_items(playlist_id, batch)


def remove_tracks_from_playlist(playlist_id, tracks, sp):
    user = sp.user("superman.fao")
    batch_size = 100  # Number of tracks to add per request (maximum is 100)
    for i in range(0, len(tracks), batch_size):
        batch = tracks[i : i + batch_size]
        sp.user_playlist_remove_all_occurrences_of_tracks(user, playlist_id, batch)


def get_date_from_user(date_str):
    while True:
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            return date
        except ValueError:
            print("Invalid date format. Please try again.")
