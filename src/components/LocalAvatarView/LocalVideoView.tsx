import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { createLocalVideoTrack } from "livekit-client";
import useResizeObserver from "use-resize-observer";
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

type Props = {
  onCanvasStreamChanged: (canvasStream: MediaStream | null) => void;
  playSfx?: (name: string) => void;
  sfxList?: string[];
};

// GlassesTransform data structure to encapsulate position, scale, rotation, and static axes
class GlassesTransform {
  position = new THREE.Vector3();
  scale = new THREE.Vector3(1, 1, 1);
  rotation = new THREE.Euler();
  upVector = new THREE.Vector3();
  sideVector = new THREE.Vector3();
  forward = new THREE.Vector3();
  // Static axes for rotation calculations
  static X_AXIS = new THREE.Vector3(1, 0, 0);
  static Y_AXIS = new THREE.Vector3(0, 1, 0);
  static Z_AXIS = new THREE.Vector3(0, 0, 1);
}

export const LocalVideoView = ({ onCanvasStreamChanged, playSfx, sfxList }: Props) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resizeRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const canvasStreamRef = useRef<MediaStream | null>(null);
  const videoTextureRef = useRef<THREE.VideoTexture | null>(null);
  const planeRef = useRef<THREE.Mesh | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const [faceLandmarkerReady, setFaceLandmarkerReady] = useState(false);
  const size = useResizeObserver({ ref: resizeRef });
  const glassesRef = useRef<THREE.Object3D | null>(null);
  const glassesLoadedRef = useRef(false);
  const glassesScaleRef = useRef(1);
  // Add refs for debug spheres
  const debugSpheresRef = useRef<{[key: string]: THREE.Mesh}>({});
  // Add refs for debug arrows
  const debugArrowsRef = useRef<{[key: string]: THREE.ArrowHelper}>({});
  const glassesTransformRef = useRef(new GlassesTransform());
  const scaleLandmarkRef = useRef<(landmark: any) => {x: number, y: number, z: number}>((landmark: any) => landmark);

  const debugColors = {
    midEyes: 0xff0000, // red
    leftEyeInnerCorner: 0x00ff00, // green
    rightEyeInnerCorner: 0x0000ff, // blue
    noseBottom: 0xffff00, // yellow
    leftEyeUpper1: 0xff00ff, // magenta
    rightEyeUpper1: 0x00ffff, // cyan
    upVector: 0xffffff, // white
    sideVector: 0x888888, // gray
    forward: 0xff8800, // orange
  };

  const animate = useRef(() => {
    
    detect();
    // Update video texture if available
    if (videoTextureRef.current) {
      videoTextureRef.current.needsUpdate = true;
    }
    // Update orbit controls
    if (controlsRef.current) {
      controlsRef.current.update();
    }
    
    rendererRef.current?.render(sceneRef.current!, cameraRef.current!);
    requestAnimationFrame(animate.current);
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
    
    // Create orthographic camera
    const aspect = size.width / size.height;
    const frustumSize = 2; // Controls the zoom level
    
    cameraRef.current = new THREE.OrthographicCamera(
      (-frustumSize * aspect) / 2,  // left
      (frustumSize * aspect) / 2,   // right
      frustumSize / 2,              // top
      -frustumSize / 2,             // bottom
      0.1,                          // near
      1000                          // far
    );
    cameraRef.current.position.z = 5;

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
      videoTextureRef.current.flipY = true;
      videoTextureRef.current.colorSpace = THREE.SRGBColorSpace;
      videoTextureRef.current.minFilter = THREE.LinearFilter;
      videoTextureRef.current.magFilter = THREE.LinearFilter;

      // Create a basic material with the video texture
      const videoMaterial = new THREE.MeshBasicMaterial({ map: videoTextureRef.current });

      // Wait for video metadata to load to get correct aspect ratio
      videoRef.current.onloadedmetadata = () => {
        const videoWidth = videoRef.current!.videoWidth;
        const videoHeight = videoRef.current!.videoHeight;
        const aspect = videoWidth / videoHeight;
        // Use a base height of 1, width = aspect
        const planeHeight = 2;
        const planeWidth = aspect * planeHeight;
        console.log("videoWidth, videoHeight", videoWidth, videoHeight);
        const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight, 128, 96);
        console.log(" planeWidth, planeHeight", planeWidth, planeHeight);
        planeRef.current = new THREE.Mesh(geometry, videoMaterial);
        // Store width/height for later use
        planeRef.current.userData.planeWidth = planeWidth;
        planeRef.current.userData.planeHeight = planeHeight;
        sceneRef.current!.add(planeRef.current);
        // Define scaleLandmark function to use latest plane size
        scaleLandmarkRef.current = (landmark: any) => ({
          x: landmark.x * planeWidth,
          y: landmark.y * planeHeight,
          z: landmark.z * planeWidth,
        });

        const loader = new GLTFLoader();
        loader.load('/3d/glasses/scene.gltf', (gltf) => {
          glassesRef.current = gltf.scene;
          // Compute scale factor for later use
          const bbox = new THREE.Box3().setFromObject(glassesRef.current);
          const sizeBox = bbox.getSize(new THREE.Vector3());
          glassesScaleRef.current = sizeBox.x;
          glassesRef.current.name = 'glasses';
          sceneRef.current!.add(glassesRef.current);
          glassesRef.current.visible = false; // Hide glasses initially
          glassesLoadedRef.current = true;
        });
      };
    }

   
    // Add debug spheres for landmarks
    // const sphereRadius = 0.03;
    // const sphereSegments = 12;
    // const sphereKeys = [
    //   'midEyes',
    //   'leftEyeInnerCorner',
    //   'rightEyeInnerCorner',
    //   'noseBottom',
    //   'leftEyeUpper1',
    //   'rightEyeUpper1',
    // ];
    // sphereKeys.forEach((key) => {
    //   const geometry = new THREE.SphereGeometry(sphereRadius, sphereSegments, sphereSegments);
    //   const material = new THREE.MeshBasicMaterial({ color: debugColors[key] });
    //   const sphere = new THREE.Mesh(geometry, material);
    //   sphere.visible = true;
    //   sceneRef.current!.add(sphere);
    //   debugSpheresRef.current[key] = sphere;
    // });

    // // Add debug arrows for upVector, sideVector, forward
    // const arrowLength = 0.3;
    // const arrowKeys = [
    //   { key: 'upVector', color: debugColors.upVector },
    //   { key: 'sideVector', color: debugColors.sideVector },
    //   { key: 'forward', color: debugColors.forward },
    // ];
    // arrowKeys.forEach(({ key, color }) => {
    //   const dir = new THREE.Vector3(1, 0, 0); // placeholder, will update
    //   const origin = new THREE.Vector3(0, 0, 0);
    //   const arrow = new THREE.ArrowHelper(dir, origin, arrowLength, color);
    //   sceneRef.current!.add(arrow);
    //   debugArrowsRef.current[key] = arrow;
    // });
  }, [size.height, size.width]);

  const setupFaceLandmarker = useCallback(async () => {
    if (typeof window === 'undefined') return;
    if (faceLandmarkerRef.current) return;

    // Load the WASM vision fileset
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
    );

    // Create the FaceLandmarker with WASM delegate
    faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU', // or 'CPU' for CPU-only
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
    setFaceLandmarkerReady(true);
  }, []);

  // Add transformLandmarks helper (ported from mediapipe-face-effects)
  const transformLandmarks = (landmarks: any) => {
    if (!landmarks) return landmarks;
    let hasVisiblity = !!landmarks.find((l: any) => l.visibility);
    let minZ = 1e-4;
    if (hasVisiblity) {
      landmarks.forEach((landmark:any) => {
        let { z, visibility } = landmark;
        z = -z;
        if (z < minZ && visibility) {
          minZ = z;
        }
      });
    } else {
      minZ = Math.max(-landmarks[234].z, -landmarks[454].z);
    }
    return landmarks.map((landmark:any) => {
      let { x, y, z } = landmark;
      return {
        x: -0.5 + x,
        y: 0.5 - y,
        z: -z - minZ,
        visibility: landmark.visibility,
      };
    });
  };

  // Detection loop
  
  const detect = () => {
    if (
      faceLandmarkerRef.current &&    
      videoRef.current
    ) {
      const startTimeMs = performance.now();
      const results = faceLandmarkerRef.current.detectForVideo(
        videoRef.current,
        startTimeMs
      );
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        // console.log(landmarks);
        // TODO: Call overlay update methods here (glasses, mask, etc.)
        // Example: glassesOverlayRef.current?.update(landmarks);
        // Example: faceMaskOverlayRef.current?.update(landmarks);
        // Example: createOrUpdateFaceMesh([landmarks]);
        updateGlassesFromLandmarks(landmarks);
      }
    }      
  };
  

  useEffect(() => {  
    createLocalVideoTrack({
      facingMode: "user",
      resolution: { 
        height: 960, 
        width: 720, 
        frameRate: 60 
        
      },
    }).then((t) => {
      t.attach(videoRef.current!);
      // Start animation loop after video is attached
      animate.current();
      setTimeout(() => {
        setupFaceLandmarker();      
      }, 2000);
    });
  }, [setupFaceLandmarker]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (!cameraRef.current) return;
    if (!size.width || !size.height) return;
    
    canvasRef.current.width = size.width + 1;
    canvasRef.current.height = size.height;
    rendererRef.current?.setSize(size.width, size.height);
    
    // Update orthographic camera frustum
    const aspect = size.width / size.height;
    const frustumSize = 2; // Keep consistent with initial setup
    
    cameraRef.current.left = (-frustumSize * aspect) / 2;
    cameraRef.current.right = (frustumSize * aspect) / 2;
    cameraRef.current.top = frustumSize / 2;
    cameraRef.current.bottom = -frustumSize / 2;
    cameraRef.current.updateProjectionMatrix();
  }, [size, size.height, size.width]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (canvasStreamRef.current) return;
    canvasStreamRef.current = canvasRef.current.captureStream(60);
    onCanvasStreamChanged(canvasStreamRef.current);
  }, [onCanvasStreamChanged]);

  useEffect(setupThreeJS, [setupThreeJS]);

  // In the detection callback, update the glasses transform
  const updateGlassesFromLandmarks = useCallback((landmarks: any) => {
    if (!glassesRef.current || !landmarks || landmarks.length < 468) return;
    // Show glasses on first valid detect
    if (!glassesRef.current.visible) {
      glassesRef.current.visible = true;
    }
    // Transform landmarks before using
    const tLandmarks = transformLandmarks(landmarks);
    // Use the scaleLandmark function defined on setup
    const scaleLandmark = scaleLandmarkRef.current;
    let midEyes = scaleLandmark(tLandmarks[168]);
    let leftEyeInnerCorner = scaleLandmark(tLandmarks[463]);
    let rightEyeInnerCorner = scaleLandmark(tLandmarks[243]);
    let noseBottom = scaleLandmark(tLandmarks[2]);
    let leftEyeUpper1 = scaleLandmark(tLandmarks[264]);
    let rightEyeUpper1 = scaleLandmark(tLandmarks[34]);

    // Use the single GlassesTransform instance
    const gt = glassesTransformRef.current;
    gt.position.set(midEyes.x, midEyes.y, midEyes.z);
    const eyeDist = Math.sqrt(
      (leftEyeUpper1.x - rightEyeUpper1.x) ** 2 +
      (leftEyeUpper1.y - rightEyeUpper1.y) ** 2 +
      (leftEyeUpper1.z - rightEyeUpper1.z) ** 2
    );
    const scale = eyeDist / glassesScaleRef.current;
    gt.scale.set(scale, scale, scale);
    gt.upVector.set(
      midEyes.x - noseBottom.x,
      midEyes.y - noseBottom.y,
      midEyes.z - noseBottom.z
    ).normalize();
    gt.sideVector.set(
      leftEyeInnerCorner.x - rightEyeInnerCorner.x,
      leftEyeInnerCorner.y - rightEyeInnerCorner.y,
      leftEyeInnerCorner.z - rightEyeInnerCorner.z
    ).normalize();
    gt.forward.crossVectors(gt.upVector, gt.sideVector).normalize();

    // Compute Euler rotation from vectors (as before)
    let zRot = (GlassesTransform.X_AXIS).angleTo(
      gt.upVector.clone().projectOnPlane(GlassesTransform.Z_AXIS)
    ) - (Math.PI / 2);
    let xRot = (Math.PI / 2) - (GlassesTransform.Z_AXIS).angleTo(
      gt.upVector.clone().projectOnPlane(GlassesTransform.X_AXIS)
    );
    let yRot = (
      new THREE.Vector3(gt.sideVector.x, 0, gt.sideVector.z)
    ).angleTo(GlassesTransform.Z_AXIS) - (Math.PI / 2);
    gt.rotation.set(xRot, yRot, zRot);

    // Update glassesRef using GlassesTransform
    glassesRef.current.position.copy(gt.position);
    glassesRef.current.scale.copy(gt.scale);
    glassesRef.current.rotation.copy(gt.rotation);

    // Update debug spheres and arrows as before
    // const debugPoints = {
    //   midEyes,
    //   leftEyeInnerCorner,
    //   rightEyeInnerCorner,
    //   noseBottom,
    //   leftEyeUpper1,
    //   rightEyeUpper1,
    // };
    // Object.entries(debugPoints).forEach(([key, pos]) => {
    //   const sphere = debugSpheresRef.current[key];
    //   if (sphere) {
    //     sphere.position.set(pos.x, pos.y, pos.z);
    //     sphere.visible = true;
    //   }
    // });
    // const arrowLength = 0.3;
    // const arrowUpdates = [
    //   { key: 'upVector', dir: gt.upVector },
    //   { key: 'sideVector', dir: gt.sideVector },
    //   { key: 'forward', dir: gt.forward },
    // ];
    // arrowUpdates.forEach(({ key, dir }) => {
    //   const arrow = debugArrowsRef.current[key];
    //   if (arrow) {
    //     arrow.position.copy(gt.position);
    //     arrow.setDirection(dir);
    //     arrow.setLength(arrowLength);
    //     arrow.visible = true;
    //   }
    // });
  }, []);

  


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
      
      {/* Control Panel */}
      {/* Remove the Lip Deformation Controls panel from the returned JSX */}
      {/* Find the <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white p-4 rounded max-w-xs"> ... </div> block with the heading 'Lip Deformation Controls' and delete it and its children. */}
      {/* SFX Debug Overlay */}
      {playSfx && sfxList && (
        <div className="fixed top-4 right-4 bg-black bg-opacity-70 text-white p-4 rounded shadow-lg z-50 max-w-xs">
          <h3 className="text-sm font-bold mb-2">SFX Debug Mixer</h3>
          <div className="space-y-2">
            {sfxList.map((sfx) => (
              <button
                key={sfx}
                onClick={() => playSfx(sfx)}
                className="block w-full bg-blue-600 hover:bg-blue-700 px-2 py-1 rounded text-xs text-left truncate"
                title={sfx}
              >
                {sfx.replace(/\.[^/.]+$/, "")}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
