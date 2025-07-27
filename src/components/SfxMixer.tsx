import { useEffect, useRef, useState, useCallback } from "react";

// Helper to fetch all sfx files in /public/sfx
async function fetchSfxList(): Promise<string[]> {
  // Hardcoded for now, could be dynamic with a manifest or server endpoint
  return ["anime-slash.mp3", "perfect.mp3"];
}

export function useSfxMixer() {
  const [sfxList, setSfxList] = useState<string[]>([]);
  const [loaded, setLoaded] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sfxBuffersRef = useRef<Record<string, AudioBuffer>>({});
  const [mixedStream, setMixedStream] = useState<MediaStream | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const destinationRef = useRef<MediaStreamAudioDestinationNode | null>(null);
  const activeSfxNodes = useRef<AudioBufferSourceNode[]>([]);
  const sfxGainRef = useRef<GainNode | null>(null);

  // Preload SFX and setup audio context
  useEffect(() => {
    let isMounted = true;
    async function setup() {
      const context = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = context;
      const sfxs = await fetchSfxList();
      setSfxList(sfxs);
      // Preload all SFX
      const buffers: Record<string, AudioBuffer> = {};
      await Promise.all(
        sfxs.map(async (name) => {
          const res = await fetch(`/sfx/${name}`);
          const arrayBuffer = await res.arrayBuffer();
          buffers[name] = await context.decodeAudioData(arrayBuffer);
        })
      );
      sfxBuffersRef.current = buffers;
      // Get mic
      const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const micSource = context.createMediaStreamSource(micStream);
      micSourceRef.current = micSource;
      // Create destination
      const destination = context.createMediaStreamDestination();
      destinationRef.current = destination;
      // Create SFX GainNode
      const sfxGain = context.createGain();
      sfxGain.gain.value = 1.0; // Default SFX volume
      sfxGainRef.current = sfxGain;
      sfxGain.connect(destination);
      // Connect mic to destination
      micSource.connect(destination);
      setMixedStream(destination.stream);
      setLoaded(true);
    }
    setup();
    return () => {
      isMounted = false;
      audioContextRef.current?.close();
    };
  }, []);

  // Play SFX by name
  const playSfx = useCallback((name: string) => {
    const context = audioContextRef.current;
    const buffer = sfxBuffersRef.current[name];
    const destination = destinationRef.current;
    if (context && buffer && destination) {
      console.log(`[SFX] Playing: ${name}`);
      context.resume(); // Ensure context is running
      const source = context.createBufferSource();
      source.buffer = buffer;
      // Connect SFX through GainNode
      if (sfxGainRef.current) {
        source.connect(sfxGainRef.current);
      } else {
        source.connect(destination);
      }
      source.start(0);
      activeSfxNodes.current.push(source);
      source.onended = () => {
        activeSfxNodes.current = activeSfxNodes.current.filter((n) => n !== source);
      };
    }
  }, []);

  return { mixedStream, playSfx, sfxList, loaded };
} 