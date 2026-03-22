import os
from livekit.api import AccessToken, VideoGrants

token = AccessToken(api_key="devkey", api_secret="secret") \
    .with_identity("user") \
    .with_name("User") \
    .with_grants(VideoGrants(room_join=True, room="test-room")) \
    .to_jwt()

print(token)