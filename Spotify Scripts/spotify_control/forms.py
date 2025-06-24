from typing import Any
from django import forms
from datetime import datetime


class SpotifyPlaylistForm(forms.Form):
    orig_playlist_link = forms.CharField(
        label="Original Playlist Link",
        widget=forms.TextInput(
            attrs={"placeholder": "Enter the original Spotify playlist link"}
        ),
    )
    new_playlist_link = forms.CharField(
        label="New Playlist Link",
        widget=forms.TextInput(
            attrs={"placeholder": "Enter the new Spotify playlist link"}
        ),
    )
    date = forms.DateTimeField(
        label="Date",
        widget=forms.DateTimeInput(
            attrs={
                "placeholder": "YYYY-MM-DD",
                "class": "form-control",
                "type": "datetime-local",
            }
        ),
    )

    def clean(self) -> dict[str, Any]:
        return super(SpotifyPlaylistForm, self).clean()
