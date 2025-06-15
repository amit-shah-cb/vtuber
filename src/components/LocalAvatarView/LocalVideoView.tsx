import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { createLocalVideoTrack } from "livekit-client";
import useResizeObserver from "use-resize-observer";

type Props = {
  onCanvasStreamChanged: (canvasStream: MediaStream | null) => void;
};

export const LocalVideoView = ({ onCanvasStreamChanged }: Props) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resizeRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const canvasStreamRef = useRef<MediaStream | null>(null);
  const videoTextureRef = useRef<THREE.VideoTexture | null>(null);
  const planeRef = useRef<THREE.Mesh | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const size = useResizeObserver({ ref: resizeRef });

  const animate = useRef(() => {
    requestAnimationFrame(animate.current);
    // Update video texture if available
    if (videoTextureRef.current) {
      videoTextureRef.current.needsUpdate = true;
    }
    // Update orbit controls
    if (controlsRef.current) {
      controlsRef.current.update();
    }
    rendererRef.current?.render(sceneRef.current!, cameraRef.current!);
  });

  const setupThreeJS = useCallback(() => {
    if (!canvasRef.current) return;
    if (sceneRef.current) return; // Already setup
    if (!size.width || !size.height) return;

    // Create scene
    sceneRef.current = new THREE.Scene();
    
    // Create renderer
    rendererRef.current = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
    });
    
    // Create camera
    cameraRef.current = new THREE.PerspectiveCamera(
      45,
      size.width / size.height,
      0.1,
      1000
    );
    cameraRef.current.position.z = 2;

    // Create orbit controls
    controlsRef.current = new OrbitControls(cameraRef.current, canvasRef.current);
    controlsRef.current.enableDamping = true;
    controlsRef.current.dampingFactor = 0.05;
    controlsRef.current.enableZoom = true;
    controlsRef.current.enableRotate = true;
    controlsRef.current.enablePan = true;

    // Add ambient light
    const light = new THREE.AmbientLight(0xffffff, 1);
    sceneRef.current.add(light);

    // Create video texture and plane when video is ready
    if (videoRef.current) {
      videoTextureRef.current = new THREE.VideoTexture(videoRef.current);
      videoTextureRef.current.colorSpace = THREE.SRGBColorSpace;
      videoTextureRef.current.minFilter = THREE.LinearFilter;
      videoTextureRef.current.magFilter = THREE.LinearFilter;

      // Create plane geometry to display video
      const geometry = new THREE.PlaneGeometry(2, 1.5);
      const material = new THREE.MeshBasicMaterial({
        map: videoTextureRef.current,
      });
      
      planeRef.current = new THREE.Mesh(geometry, material);
      sceneRef.current.add(planeRef.current);
    }
  }, [size.height, size.width]);

  useEffect(() => {
    createLocalVideoTrack({
      facingMode: "user",
      resolution: { width: 1080, height: 1920, frameRate: 30 },
    }).then((t) => {
      t.attach(videoRef.current!);
      // Start animation loop after video is attached
      animate.current();
    });
  }, []);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (!cameraRef.current) return;
    if (!size.width || !size.height) return;
    
    canvasRef.current.width = size.width + 1;
    canvasRef.current.height = size.height;
    rendererRef.current?.setSize(size.width, size.height);
    cameraRef.current.aspect = size.width / size.height;
    cameraRef.current.updateProjectionMatrix();
  }, [size, size.height, size.width]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (canvasStreamRef.current) return;
    canvasStreamRef.current = canvasRef.current.captureStream(60);
    onCanvasStreamChanged(canvasStreamRef.current);
  }, [onCanvasStreamChanged]);

  useEffect(setupThreeJS, [setupThreeJS]);

  return (
    <div className="relative h-full w-full">
      <div className="overflow-hidden h-full" ref={resizeRef}>
        <canvas
          width={size.width}
          height={size.height}
          className="h-full w-full"
          ref={canvasRef}
        />
      </div>
      <div className="absolute w-[0px] h-[0px] bottom-2 right-2 overflow-hidden">
        <video className="h-full w-full" ref={videoRef} />
      </div>
    </div>
  );
};
