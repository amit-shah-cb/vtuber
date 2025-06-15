import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { createLocalVideoTrack } from "livekit-client";
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
  const size = useResizeObserver({ ref: resizeRef });

  // Official MediaPipe face mesh indices for specific facial features
  const faceIndices = useRef<number[]>([]);

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
      const x = (0.5 - landmark.x) * 2;        // Flip X back to match video movement direction
      const y = (0.5 - landmark.y) * 1.5;      // Flip Y to match video texture and scale
      const z = landmark.z * 0.5 || 0;         // Scale Z depth
      
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
      videoTextureRef.current.flipY = true;
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
      resolution: { 
        width: 1080, 
        height: 1920, 
        frameRate: 30 
      },
    }).then((t) => {
      t.attach(videoRef.current!);
      // Start animation loop after video is attached
      animate.current();
      
      // Setup FaceLandmarker after video is ready
      setTimeout(() => {
        setupFaceMesh();
      }, 2000);
    });
  }, [setupFaceMesh]);

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
