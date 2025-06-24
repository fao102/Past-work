import spotipy
from spotipy.oauth2 import SpotifyOAuth
from datetime import datetime
import re
import base64
import requests

# Authenticate with Spotify


def create_spotify():
    auth_manager = SpotifyOAuth(
        client_id="94966b3f99094e03892a3bee1d35e3c4",
        client_secret="8379f65c85b44f3a894aab704d98f6f2",
        redirect_uri="http://127.0.0.1:5500/",
        scope="playlist-modify-public",
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)

    return auth_manager, sp


def refresh_spotify(auth_manager, spotify):
    token_info = auth_manager.cache_handler.get_cached_token()
    if auth_manager.is_token_expired(token_info):
        auth_manager, spotify = create_spotify()
    return auth_manager, spotify


# Function to get playlist ID from playlist link
def get_playlist_id(playlist_link):
    # Extract playlist ID from the link
    match = re.search(r"playlist/([^/?]+)", playlist_link)
    if match:
        return match.group(1)
    else:
        return None


# Function to get all tracks from a playlist
def get_all_tracks(orig_playlist_id):
    tracks = []
    offset = 0
    while True:
        response = sp.playlist_tracks(orig_playlist_id, offset=offset)
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

        added_at = datetime.strptime(track["added_at"].split("T")[0], "%Y-%m-%d")

        if added_at > date:
            filtered_tracks.append(track["track"]["uri"])
    return filtered_tracks


# Function to filter tracks added in 2020
def filter_tracks_by_date(tracks, date):
    filtered_tracks = []
    for track in tracks:

        added_at = datetime.strptime(track["added_at"].split("T")[0], "%Y-%m-%d")

        if added_at > date:
            filtered_tracks.append(track)
    return filtered_tracks


def get_date_from_user(date_str):
    while True:
        try:
            date = datetime.strptime(date_str, "%d-%m-%Y")
            return date
        except ValueError:
            print("Invalid date format. Please try again.")


def add_tracks_to_playlist(playlist_id, tracks):
    batch_size = 100  # Number of tracks to add per request (maximum is 100)
    for i in range(0, len(tracks), batch_size):
        batch = tracks[i : i + batch_size]
        sp.playlist_add_items(playlist_id, batch)


def remove_tracks_from_playlist(playlist_id, tracks):
    batch_size = 100  # Number of tracks to add per request (maximum is 100)
    for i in range(0, len(tracks), batch_size):
        batch = tracks[i : i + batch_size]
        sp.user_playlist_remove_all_occurrences_of_tracks(user, playlist_id, batch)


def main():
    # Get playlist link from the user

    orig_playlist_link = input("Enter the original Spotify playlist link: ")
    new_playlist_link = input("Enter the new Spotify playlist link: ")
    date = get_date_from_user(input("Please enter the date (YYYY-MM-DD): "))

    # Get playlist ID from the link
    orig_playlist_id = get_playlist_id(orig_playlist_link)
    new_playlist_id = get_playlist_id(new_playlist_link)

    if orig_playlist_id:
        # Get all tracks from the playlist
        all_tracks = get_all_tracks(orig_playlist_id)

        # Filter tracks added in 2020
        filtered_track_uris = filter_tracks_uri_by_date(all_tracks, date)
        print("These tracks will be removed from the playlist,  ")

        filtered_tracks = filter_tracks_by_date(all_tracks, date)
        for track in filtered_tracks:
            print(track["track"]["name"], "-", track["track"]["artists"][0]["name"])

        decision = input("Are you sure you want to continue? Y/N ")

        if decision.upper() == "Y":

            add_tracks_to_playlist(new_playlist_id, filtered_track_uris)
            remove_tracks_from_playlist(orig_playlist_id, filtered_track_uris)

            print(f"{len(filtered_track_uris)} tracks have been removed !")
        else:
            exit()

    else:
        print("Invalid playlist link. Please provide a valid Spotify playlist link.")


auth_manager, sp = create_spotify()


main()
