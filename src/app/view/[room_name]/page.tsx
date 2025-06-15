"use client";

import {
  LiveKitRoom,
  ParticipantAudioTile,
  ParticipantContextIfNeeded,
  ParticipantTile,
  useMediaTrack,
  useRemoteParticipant,
} from "@livekit/components-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { ConnectionDetails } from "@/pages/api/connection_details";
import { Track } from "livekit-client";

export default function Page({ params }: any) {
  const [connectionDetails, setConnectionDetails] =
    useState<ConnectionDetails | null>(null);
  const roomName = (params.room_name || "") as string;

  const requestConnectionDetails = useCallback(async () => {
    const response = await fetch("/api/connection_details_viewer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room: roomName }),
    });

    if (response.status === 200) {
      return response.json();
    }

    const { error } = await response.json();
    throw error;
  }, [roomName]);

  useEffect(() => {
    requestConnectionDetails().then(setConnectionDetails);
  }, [requestConnectionDetails]);

  if (!connectionDetails) {
    return null;
  }

  // Show the room UI
  return (
    <div className="flex items-center justify-center w-screen h-screen">
      <LiveKitRoom
        token={connectionDetails.token}
        serverUrl={connectionDetails.ws_url}
        connect={true}
      >
        <StreamerTile />
      </LiveKitRoom>
    </div>
  );
}

const StreamerTile = () => {
  const part = useRemoteParticipant("streamer");
  const videoRef = useRef<HTMLVideoElement | null>(null);

  if (!part) return null;

  return (
    <ParticipantContextIfNeeded participant={part}>
      <Video />
    </ParticipantContextIfNeeded>
  );
};

const Video = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoTrack = useMediaTrack(Track.Source.Camera);
  const audioTrack = useMediaTrack(Track.Source.Microphone);
  const [isMuted, setIsMuted] = useState(true);

  useEffect(() => {
    const refCurrent = videoRef.current;
    if (videoTrack) {
      console.log("attaching video track");
      videoTrack.track?.attach(videoRef.current!);
    }
    if (audioTrack) {
      console.log("attaching audio track");
      audioTrack.track?.attach(videoRef.current!);
    }
    return () => {
      if (videoTrack) {
        videoTrack.track?.detach(refCurrent!);
      }
      if (audioTrack) {
        audioTrack.track?.detach(refCurrent!);
      }
    };
  }, [audioTrack, videoTrack]);

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  return (
    <div className="relative w-full h-full">
      <video 
        ref={videoRef}       
      />
      <button
        onClick={toggleMute}
        className="absolute top-4 right-4 bg-black bg-opacity-50 text-white px-4 py-2 rounded hover:bg-opacity-70 transition-opacity"
      >
        {isMuted ? "🔇 Unmute" : "🔊 Mute"}
      </button>
    </div>
  );
};
