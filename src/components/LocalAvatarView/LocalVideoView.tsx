import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { createLocalVideoTrack, LocalVideoTrack } from "livekit-client";
import useResizeObserver from "use-resize-observer";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { 
  FACEMESH_LEFT_EYE, 
  FACEMESH_RIGHT_EYE, 
  FACEMESH_LIPS, 
  FACEMESH_LEFT_EYEBROW, 
  FACEMESH_RIGHT_EYEBROW, 
  FACEMESH_FACE_OVAL 
} from "@mediapipe/face_mesh";

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
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const faceMeshRef = useRef<THREE.LineSegments | null>(null);
  const faceGeometryRef = useRef<THREE.BufferGeometry | null>(null);
  const faceMaterialRef = useRef<THREE.LineBasicMaterial | null>(null);
  const videoTrackRef = useRef<LocalVideoTrack | null>(null);
  const size = useResizeObserver({ ref: resizeRef });
  
  // Standard video configuration - always capture in 4:3 landscape
  const VIDEO_WIDTH = 1024;
  const VIDEO_HEIGHT = 768;
  const VIDEO_ASPECT = VIDEO_WIDTH / VIDEO_HEIGHT; // 4:3 aspect ratio
  const PLANE_WIDTH = 2;
  const PLANE_HEIGHT = 1.5; // 4:3 aspect ratio to match video

  // Orientation state
  const [orientation, setOrientation] = useState<"portrait" | "landscape">("portrait");

  // Official MediaPipe face mesh indices for specific facial features
  const faceIndices = useRef<number[]>([]);

  // Orientation detection
  const detectOrientation = useCallback(() => {
    if (typeof window === 'undefined') return;
    
    // Use screen orientation API if available
    if (screen.orientation) {
      const angle = screen.orientation.angle;
      const newOrientation = (angle === 0 || angle === 180) ? "portrait" : "landscape";
      setOrientation(newOrientation);
    } else {
      // Fallback to window dimensions
      const isPortrait = window.innerHeight > window.innerWidth;
      setOrientation(isPortrait ? "portrait" : "landscape");
    }
  }, []);

  // Adjust camera position based on orientation
  const adjustCameraForOrientation = useCallback(() => {
    if (!cameraRef.current) return;
    
    if (orientation === "portrait") {
      // Move camera closer for portrait to show more of the face
      cameraRef.current.position.z = 1.5;
      cameraRef.current.fov = 50;
    } else {
      // Move camera back for landscape to show full frame
      cameraRef.current.position.z = 2;
      cameraRef.current.fov = 45;
    }
    
    cameraRef.current.updateProjectionMatrix();
  }, [orientation]);

  // Initialize face indices with official MediaPipe facial feature data
  useEffect(() => {
    const indices: number[] = [];
    
    // Combine all facial feature edges into one array
    const allFacialFeatures = [
      ...FACEMESH_FACE_OVAL,      // Face outline
      ...FACEMESH_LEFT_EYE,       // Left eye
      ...FACEMESH_RIGHT_EYE,      // Right eye  
      ...FACEMESH_LIPS,           // Mouth/lips
      ...FACEMESH_LEFT_EYEBROW,   // Left eyebrow
      ...FACEMESH_RIGHT_EYEBROW   // Right eyebrow
    ];
    
    allFacialFeatures.forEach((edge) => {
      // Each edge is a pair of vertex indices [from, to]
      indices.push(edge[0], edge[1]);
    });
    
    faceIndices.current = indices;
    console.log(`Loaded ${allFacialFeatures.length} edges from official MediaPipe facial features`);
    console.log(`Face oval: ${FACEMESH_FACE_OVAL.length}, Eyes: ${FACEMESH_LEFT_EYE.length + FACEMESH_RIGHT_EYE.length}, Lips: ${FACEMESH_LIPS.length}, Eyebrows: ${FACEMESH_LEFT_EYEBROW.length + FACEMESH_RIGHT_EYEBROW.length}`);
  }, []);

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

  const initializeFaceMesh = useCallback(() => {
    if (faceGeometryRef.current && faceMaterialRef.current) return; // Already initialized

    // Create geometry once
    faceGeometryRef.current = new THREE.BufferGeometry();
    
    // Create line material for wireframe edges
    faceMaterialRef.current = new THREE.LineBasicMaterial({
      color: 0x00ff00,
      transparent: true,
      opacity: 0.8,
      linewidth: 2
    });

    // Set indices once (they don't change)
    faceGeometryRef.current.setIndex(faceIndices.current);
    
    // Create initial empty attributes (will be updated later)
    const initialVertices = new Float32Array(468 * 3); // 468 landmarks * 3 coordinates
    
    faceGeometryRef.current.setAttribute('position', new THREE.BufferAttribute(initialVertices, 3));
  }, []);

  const updateFaceMesh = useCallback((landmarks: any[]) => {
    if (!faceGeometryRef.current || !faceMaterialRef.current || !landmarks || landmarks.length === 0) return;
    
    const vertices = faceGeometryRef.current.attributes.position.array as Float32Array;
    
    landmarks.forEach((landmark, index) => {
      // Convert normalized coordinates to world space
      // Since mesh is rotated 180° around Y-axis, flip X coordinate to match movement direction
      const x = (0.5 - landmark.x) * PLANE_WIDTH;   // Flip X back to match video movement direction
      const y = (0.5 - landmark.y) * PLANE_HEIGHT;  // Flip Y to match video texture and scale
      const z = landmark.z * 0.5 || 0;              // Scale Z depth
      
      vertices[index * 3] = x;
      vertices[index * 3 + 1] = y;
      vertices[index * 3 + 2] = z;
    });
    
    // Mark attributes as needing update
    faceGeometryRef.current.attributes.position.needsUpdate = true;
  }, []);

  const createOrUpdateFaceMesh = useCallback((faceLandmarks: any[]) => {
    if (!sceneRef.current || !faceLandmarks || faceLandmarks.length === 0) return;
    
    // Initialize geometry and material if not done already
    initializeFaceMesh();
    
    // Update the mesh with new landmark data
    updateFaceMesh(faceLandmarks[0]);
    
    // Create line segments if it doesn't exist, otherwise just update existing one
    if (!faceMeshRef.current) {
      faceMeshRef.current = new THREE.LineSegments(faceGeometryRef.current!, faceMaterialRef.current!);
      faceMeshRef.current.position.z = 0.02; // Slightly in front of video plane
      
      // Rotate 180 degrees around Y-axis so face mesh faces same direction as face in video
      faceMeshRef.current.rotation.y = Math.PI; // 180 degrees rotation
      
      sceneRef.current.add(faceMeshRef.current);
    } else {
      // Optional: Add continuous rotation animation
      // Uncomment the line below for animated rotation
      // faceMeshRef.current.rotation.y += 0.01;
    }
  }, [initializeFaceMesh, updateFaceMesh]);

  const removeFaceMesh = useCallback(() => {
    if (faceMeshRef.current && sceneRef.current) {
      sceneRef.current.remove(faceMeshRef.current);
      faceMeshRef.current = null;
    }
  }, []);

  const setupFaceMesh = useCallback(async () => {
    // Ensure we're running on client side
    if (typeof window === 'undefined') return;
    
    try {
      console.log("Initializing FaceLandmarker...");
      
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
      );
      
      console.log("Vision tasks initialized");
      
      // Try GPU first, fallback to CPU if it fails
      try {
        faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false
        });
        console.log("FaceLandmarker model loaded with GPU acceleration");
      } catch (gpuError) {
        console.warn("GPU initialization failed, falling back to CPU:", gpuError);
        
        faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "CPU"
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false
        });
        console.log("FaceLandmarker model loaded with CPU");
      }
      
      // Start detection loop
      const detectFaceMesh = () => {
        if (faceLandmarkerRef.current && videoRef.current && videoRef.current.videoWidth > 0) {
          try {
            const startTimeMs = performance.now();
            const results = faceLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
            
            if (results.faceLandmarks && results.faceLandmarks.length > 0) {
              createOrUpdateFaceMesh(results.faceLandmarks);
            } else {
              // Remove face mesh if no faces found
              removeFaceMesh();
            }
          } catch (detectionError) {
            console.warn("Face mesh detection error:", detectionError);
          }
        }
        requestAnimationFrame(detectFaceMesh);
      };
      
      // Wait for video to be fully ready
      setTimeout(() => {
        detectFaceMesh();
      }, 500);
      
    } catch (error) {
      console.error("Error setting up FaceLandmarker:", error);
      // Retry after a delay
      setTimeout(() => {
        console.log("Retrying FaceLandmarker setup...");
        setupFaceMesh();
      }, 2000);
    }
  }, [createOrUpdateFaceMesh, removeFaceMesh]);

  const updateVideoPlane = useCallback(() => {
    if (!sceneRef.current || !videoRef.current || !videoTextureRef.current || !size.width || !size.height) return;

    // Remove existing plane if it exists
    if (planeRef.current) {
      sceneRef.current.remove(planeRef.current);
      planeRef.current.geometry.dispose();
      
      // Handle both single material and material array
      if (Array.isArray(planeRef.current.material)) {
        planeRef.current.material.forEach(material => material.dispose());
      } else {
        planeRef.current.material.dispose();
      }
    }

    // Create plane geometry with standard 4:3 aspect ratio
    const geometry = new THREE.PlaneGeometry(PLANE_WIDTH, PLANE_HEIGHT);
    const material = new THREE.MeshBasicMaterial({
      map: videoTextureRef.current,
    });
    
    planeRef.current = new THREE.Mesh(geometry, material);
    sceneRef.current.add(planeRef.current);
  }, [size.height, size.width]);

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
    
    // Apply orientation-based camera adjustments
    adjustCameraForOrientation();

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

    // Create video texture when video is ready
    if (videoRef.current) {
      videoTextureRef.current = new THREE.VideoTexture(videoRef.current);
      videoTextureRef.current.flipY = true;
      videoTextureRef.current.colorSpace = THREE.SRGBColorSpace;
      videoTextureRef.current.minFilter = THREE.LinearFilter;
      videoTextureRef.current.magFilter = THREE.LinearFilter;
      
      // Create the video plane
      updateVideoPlane();
    }
  }, [size.height, size.width, updateVideoPlane, adjustCameraForOrientation]);

  // Create video track with standard resolution
  const createVideoTrack = useCallback(async () => {
    // Stop existing track if it exists
    if (videoTrackRef.current) {
      videoTrackRef.current.stop();
      videoTrackRef.current = null;
    }

    try {
      const track = await createLocalVideoTrack({
        facingMode: "user",
        resolution: { 
          width: VIDEO_WIDTH, 
          height: VIDEO_HEIGHT, 
          frameRate: 30 
        },
      });
      
      videoTrackRef.current = track;
      track.attach(videoRef.current!);
      
      // Start animation loop after video is attached
      animate.current();
      
      // Setup FaceLandmarker after video is ready
      setTimeout(() => {
        setupFaceMesh();
      }, 2000);
    } catch (error) {
      console.error("Error creating video track:", error);
    }
  }, [setupFaceMesh]);

  useEffect(() => {  
    createVideoTrack();
  }, [createVideoTrack]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (!cameraRef.current) return;
    if (!size.width || !size.height) return;
    
    // Always use a fixed canvas resolution that maintains 4:3 aspect ratio
    // Let CSS handle the display sizing
    const canvasWidth = 800;  // Fixed width
    const canvasHeight = 600; // Fixed height (4:3 ratio)
    
    canvasRef.current.width = canvasWidth;
    canvasRef.current.height = canvasHeight;
    
    rendererRef.current?.setSize(canvasWidth, canvasHeight);
    cameraRef.current.aspect = canvasWidth / canvasHeight;
    cameraRef.current.updateProjectionMatrix();
  }, [size, size.height, size.width]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (canvasStreamRef.current) return;
    canvasStreamRef.current = canvasRef.current.captureStream(60);
    onCanvasStreamChanged(canvasStreamRef.current);
  }, [onCanvasStreamChanged]);

  useEffect(setupThreeJS, [setupThreeJS]);

  // Update video plane when size changes
  useEffect(() => {
    if (sceneRef.current && videoTextureRef.current) {
      updateVideoPlane();
    }
  }, [updateVideoPlane]);

  // Orientation detection and handling
  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    // Initial orientation detection
    detectOrientation();
    
    // Listen for orientation changes
    const handleOrientationChange = () => {
      setTimeout(() => {
        detectOrientation();
      }, 100); // Small delay to ensure screen dimensions are updated
    };
    
    // Listen for various orientation change events
    window.addEventListener('orientationchange', handleOrientationChange);
    window.addEventListener('resize', handleOrientationChange);
    
    // Listen for screen orientation API if available
    if (screen.orientation) {
      screen.orientation.addEventListener('change', handleOrientationChange);
    }
    
    return () => {
      window.removeEventListener('orientationchange', handleOrientationChange);
      window.removeEventListener('resize', handleOrientationChange);
      if (screen.orientation) {
        screen.orientation.removeEventListener('change', handleOrientationChange);
      }
    };
  }, [detectOrientation]);

  // Adjust camera when orientation changes
  useEffect(() => {
    adjustCameraForOrientation();
  }, [orientation, adjustCameraForOrientation]);

  // Cleanup video track on unmount
  useEffect(() => {
    return () => {
      if (videoTrackRef.current) {
        videoTrackRef.current.stop();
        videoTrackRef.current = null;
      }
    };
  }, []);

  return (
    <div className="relative h-full w-full flex items-center justify-center bg-black" ref={resizeRef}>
      <canvas
        className="max-h-full max-w-full object-contain"
        style={{ aspectRatio: '4/3' }}
        ref={canvasRef}
      />
      <div className="absolute w-[0px] h-[0px] bottom-2 right-2 overflow-hidden">
        <video className="h-full w-full" ref={videoRef} />
      </div>
    </div>
  );
};
