import { BroadcastDetails } from "@/pages/api/broadcast";
import {
  useConnectionState,
  useLiveKitRoom,
  useLocalParticipant,
  useMediaTrack,
  useRoomInfo,
  useTracks,
} from "@livekit/components-react";
import { ConnectionState, Track } from "livekit-client";
import { useCallback, useMemo, useState } from "react";
import { EgressDestination } from "./EgressDestination";
import { LocalVideoView } from "./LocalAvatarView/LocalVideoView";

export function MeetView() {
  const connectionState = useConnectionState();
  const { name } = useRoomInfo();
  const { localParticipant } = useLocalParticipant();
  const [canvasStream, setCavasStream] = useState<MediaStream | null>(null);
  const [broadcastLoading, setBroadcastLoading] = useState(false);
  const { track: micTrack } = useMediaTrack(
    Track.Source.Microphone,
    localParticipant
  );
  const [isLive, setIsLive] = useState(false);
  const [isDirty, setIsDirty] = useState(false);

  // StreamKeys
  const [twitchEnabled, setTwitchEnabled] = useState(false);
  const [twitchStreamKey, setTwitchStreamKey] = useState("");
  const [youtubeEnabled, setYouTubeEnabled] = useState(false);
  const [youtubeStreamKey, setYouTubeStreamKey] = useState("");

  const stopBroadcast = useCallback(async () => {
    setBroadcastLoading(true);
    try {
      const publishedTracks = localParticipant.getTracks();
      const tracks = publishedTracks
        .map((t) => t.track)
        .filter((t) => t)
        .map((t) => t?.mediaStreamTrack!);
      await localParticipant.unpublishTracks(tracks);
      setIsLive(false);
      setIsDirty(false);
    } catch (e) {
      console.log(e);
    } finally {
      setBroadcastLoading(false);
    }
  }, [localParticipant]);

  const broadcast = useCallback(async () => {
    setBroadcastLoading(true);
    const body: BroadcastDetails = {
      room_name: name,
      twitch_stream_key: twitchEnabled ? twitchStreamKey : undefined,
    };

    try {
      const track = canvasStream!.getTracks()[0];
      await localParticipant.publishTrack(track, {
        source: Track.Source.Camera,
      });
      const mic = micTrack?.mediaStream?.getTracks()[0];
      if (mic) {
        await localParticipant.publishTrack(mic);
      }
      await fetch("/api/broadcast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setIsLive(true);
    } catch (e) {
      const publishedTracks = localParticipant.getTracks();
      const tracks = publishedTracks
        .map((t) => t.track)
        .filter((t) => t)
        .map((t) => t?.mediaStreamTrack!);
      await localParticipant.unpublishTracks(tracks);
      throw e;
    } finally {
      setBroadcastLoading(false);
    }
  }, [
    canvasStream,
    localParticipant,
    micTrack,
    name,
    twitchEnabled,
    twitchStreamKey,
  ]);

  const viewerLink = useMemo(() => {
    if (typeof window === "undefined") {
      return "";
    }
    return `${window.location.origin}/view/${name}`;
  }, [name]);

  const broadcastButtonText = useMemo(() => {
    if (broadcastLoading) {
      return "";
    }
    return isLive ? "Stop Stream" : "Go Live";
  }, [broadcastLoading, isLive]);

  const setEnabled = useCallback(
    (type: "twitch" | "youtube") => (enabled: boolean) => {
      if (isLive) {
        setIsDirty(true);
      }
      if (type === "twitch") {
        setTwitchEnabled(enabled);
      } else if (type === "youtube") {
        setYouTubeEnabled(enabled);
      }
    },
    [isLive]
  );

  const setStreamKey = useCallback(
    (type: "twitch" | "youtube") => (key: string) => {
      if (isLive) {
        setIsDirty(true);
      }
      if (type === "twitch") {
        setTwitchStreamKey(key);
      } else if (type === "youtube") {
        setYouTubeStreamKey(key);
      }
    },
    [isLive]
  );

  if (connectionState !== ConnectionState.Connected) {
    return null;
  }

  return (
    <div className="relative flex h-full w-full">
      <div className="h-[100vh] w-[100vw]">
        <LocalVideoView
          onCanvasStreamChanged={(ms) => {
            setCavasStream(ms);
          }}
        />
      </div>
      
      {/* Floating Bottom Bar */}
      <div className="absolute bottom-4 left-4 right-4 bg-black bg-opacity-80 backdrop-blur-sm rounded-lg p-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <a
            className="text-white hover:text-blue-400 transition-colors underline"
            target="_blank"
            rel="noreferrer"
            href={viewerLink}
          >
            Preview Link
          </a>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            className={`px-6 py-2 rounded-md font-medium transition-colors ${
              isLive 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-green-600 hover:bg-green-700 text-white'
            } ${broadcastLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={async () => {
              if (broadcastLoading) return;
              if (isLive) {
                await stopBroadcast();
              } else {
                await broadcast();
              }
            }}
            disabled={broadcastLoading}
          >
            {broadcastLoading ? 'Loading...' : broadcastButtonText}
          </button>
        </div>
      </div>
    </div>
  );
}
