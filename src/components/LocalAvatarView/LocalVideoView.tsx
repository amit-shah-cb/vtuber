import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { createLocalVideoTrack, LocalVideoTrack } from "livekit-client";
import useResizeObserver from "use-resize-observer";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { useMobile } from "@/util/useMobile";
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
  const faceBoundingBoxRef = useRef<THREE.LineSegments | null>(null);
  const videoTrackRef = useRef<LocalVideoTrack | null>(null);
  const size = useResizeObserver({ ref: resizeRef });
  const isMobile = useMobile();
  
  // Get video resolution based on device type - Facebook Live 720p streaming specs
  const getVideoResolution = useCallback(() => {
    if (isMobile) {
      return { width: 540, height: 720 }; // 3:4 aspect ratio (portrait for mobile) - 720p height
    } else {
      return { width: 960, height: 720 }; // 4:3 aspect ratio (landscape for desktop) - 720p height
    }
  }, [isMobile]);

  // Update plane aspect ratio based on actual video metadata
  const updatePlaneAspect = useCallback(() => {
    console.log('🔄 CALLBACK: updatePlaneAspect called');
    if (!videoRef.current || !sceneRef.current || !videoTextureRef.current) return;
    
    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;
    
    if (videoWidth === 0 || videoHeight === 0) return; // Video not loaded yet
    
    const videoAspect = videoWidth / videoHeight;
    
    // Debug: Log quality-related information
    console.log(`🎥 VIDEO QUALITY DEBUG:`);
    console.log(`  Resolution: ${videoWidth}x${videoHeight}`);
    console.log(`  Aspect: ${videoAspect.toFixed(3)}`);
    console.log(`  Canvas: ${canvasRef.current?.width}x${canvasRef.current?.height}`);
    console.log(`  Container: ${size.width}x${size.height}`);
    if (videoTrackRef.current) {
      console.log(`  Track settings:`, videoTrackRef.current.mediaStreamTrack.getSettings());
    }
    
    // Calculate plane dimensions maintaining aspect ratio
    // Use a base size of 2 units and scale appropriately
    let planeWidth, planeHeight;
    
    if (videoAspect > 1) {
      // Landscape video (width > height)
      planeWidth = 2;
      planeHeight = 2 / videoAspect;
    } else {
      // Portrait video (height > width)
      planeHeight = 2;
      planeWidth = 2 * videoAspect;
    }
    
    console.log(`📐 Updating plane: ${planeWidth.toFixed(3)}x${planeHeight.toFixed(3)}`);
    
    // Update the plane geometry
    if (planeRef.current) {
      sceneRef.current.remove(planeRef.current);
      planeRef.current.geometry.dispose();
      
      // Handle both single material and material array
      if (Array.isArray(planeRef.current.material)) {
        planeRef.current.material.forEach(material => material.dispose());
      } else {
        planeRef.current.material.dispose();
      }
      
      console.log(`♻️ Recreated plane and material for quality improvement`);
    }
    
    // Create new plane with correct aspect ratio
    const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
    const material = new THREE.MeshBasicMaterial({
      map: videoTextureRef.current,
    });
    
    planeRef.current = new THREE.Mesh(geometry, material);
    sceneRef.current.add(planeRef.current);
    
    console.log(`✅ Plane aspect update complete`);
    
    return { width: planeWidth, height: planeHeight };
  }, [size.width, size.height]);

  // Get plane dimensions (fallback for face mesh when video isn't loaded)
  const getPlaneDimensions = useCallback(() => {
    if (!videoRef.current || videoRef.current.videoWidth === 0) {
      // Fallback dimensions
      if (isMobile) {
        return { width: 1.5, height: 2 }; // 3:4 aspect ratio (portrait for mobile)
      } else {
        return { width: 2, height: 1.5 }; // 4:3 aspect ratio (landscape for desktop)
      }
    }
    
    // Use actual video dimensions
    const videoAspect = videoRef.current.videoWidth / videoRef.current.videoHeight;
    if (videoAspect > 1) {
      // Landscape
      return { width: 2, height: 2 / videoAspect };
    } else {
      // Portrait
      return { width: 2 * videoAspect, height: 2 };
    }
  }, [isMobile]);

  // Official MediaPipe face mesh indices for specific facial features
  const faceIndices = useRef<number[]>([]);

  // Adjust camera position based on device type
  const adjustCameraForDevice = useCallback(() => {
    console.log('🔄 CALLBACK: adjustCameraForDevice called', {isMobile});
    if (!cameraRef.current) return;
    
    if (isMobile) {
      // Move camera closer for mobile to show more of the face
      cameraRef.current.position.z = 1.5;
    } else {
      // Move camera back for desktop to show full frame
      cameraRef.current.position.z = 2;
    }
    
    cameraRef.current.updateProjectionMatrix();
  }, [isMobile]);

  // Initialize face indices with official MediaPipe facial feature data
  useEffect(() => {
    console.log('🎯 EFFECT [1/9]: Face indices initialization (no deps)');
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
    const planeDimensions = getPlaneDimensions();
    
    landmarks.forEach((landmark, index) => {
      // Convert normalized coordinates to world space
      // Since mesh is rotated 180° around Y-axis, flip X coordinate to match movement direction
      const x = (0.5 - landmark.x) * planeDimensions.width;   // Flip X back to match video movement direction
      const y = (0.5 - landmark.y) * planeDimensions.height;  // Flip Y to match video texture and scale
      const z = landmark.z * 0.5 || 0;                        // Scale Z depth
      
      vertices[index * 3] = x;
      vertices[index * 3 + 1] = y;
      vertices[index * 3 + 2] = z;
    });
    
    // Mark attributes as needing update
    faceGeometryRef.current.attributes.position.needsUpdate = true;
  }, [getPlaneDimensions]);

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
              // createOrUpdateFaceMesh(results.faceLandmarks);
              createOrUpdateFaceBoundingBox(results.faceLandmarks);
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


  const createOrUpdateFaceBoundingBox = useCallback((faceLandmarks: any[]) => {
    if (!sceneRef.current || !faceLandmarks || faceLandmarks.length === 0) {
      console.log('createOrUpdateFaceBoundingBox: Missing scene or landmarks');
      return;
    }
    
    console.log('createOrUpdateFaceBoundingBox: Processing', faceLandmarks.length, 'face(s)');
    
    const landmarks = faceLandmarks[0];
    
    // Calculate bounding box from face landmarks
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    landmarks.forEach((landmark: any) => {
      // Convert normalized coordinates to world space (same as face mesh)
      const x = (landmark.x - 0.5) * 2;        // Convert to -1 to +1 range (NOT flipped)
      const y = (0.5 - landmark.y) * 1.5;      // Flip Y and convert to -0.75 to +0.75 range
      const z = landmark.z * 0.5 || 0;         // Scale Z depth
      
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    });
    
    // Calculate bounding box dimensions and center
    const width = maxX - minX;
    const height = maxY - minY;
    const depth = maxZ - minZ;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    
    console.log('Face Bounding Box calculated:', { width, height, centerX, centerY, centerZ });
    
    // Create or update bounding box plane
    if (!faceBoundingBoxRef.current) {
      console.log('Creating new face bounding box outline');
      
      // Create outline geometry using line segments
      const outlineGeometry = new THREE.BufferGeometry();
      
      // Define the vertices for a rectangle outline
      const vertices = new Float32Array([
        -0.5, -0.5, 0,  // Bottom left
         0.5, -0.5, 0,  // Bottom right
         0.5,  0.5, 0,  // Top right
        -0.5,  0.5, 0   // Top left
      ]);
      
      // Define indices to connect the vertices into a rectangle outline
      const indices = [
        0, 1,  // Bottom edge
        1, 2,  // Right edge
        2, 3,  // Top edge
        3, 0   // Left edge
      ];
      
      outlineGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
      outlineGeometry.setIndex(indices);
      
      const outlineMaterial = new THREE.LineBasicMaterial({
        color: 0x00ffff,
        linewidth: 2,
        transparent: true,
        opacity: 0.8
      });
      
      faceBoundingBoxRef.current = new THREE.LineSegments(outlineGeometry, outlineMaterial);
      sceneRef.current.add(faceBoundingBoxRef.current);
      console.log('Face bounding box outline created and added to scene');
    }
    
    // Update bounding box size and position
    faceBoundingBoxRef.current.scale.set(width, height, 1);
    faceBoundingBoxRef.current.position.set(centerX, centerY, 0.1);
    console.log('Face bounding box updated - scale:', width, height, 'position:', centerX, centerY, 0.1);
    
    // Create or update normal vector
    // if (!faceNormalVectorRef.current) {
    //   const direction = new THREE.Vector3(0, 0, 1);
    //   const origin = new THREE.Vector3();
    //   const length = Math.max(width, height) * 0.5;
      
    //   faceNormalVectorRef.current = new THREE.ArrowHelper(
    //     direction,
    //     origin,
    //     length,
    //     0xff0000, // Red color
    //     length * 0.2,
    //     length * 0.1
    //   );
      
    //   sceneRef.current.add(faceNormalVectorRef.current);
    //   console.log('Face normal vector created and added to scene');
    // }
    
    // // Update normal vector position and size
    // const normalLength = Math.max(width, height) * 0.5;
    // faceNormalVectorRef.current.position.set(centerX, centerY, 0.15);
    // faceNormalVectorRef.current.setLength(normalLength, normalLength * 0.2, normalLength * 0.1);
    // console.log('Face normal vector updated - position:', centerX, centerY, 0.15);
  }, []);

  const removeFaceBoundingBox = useCallback(() => {
    if (faceBoundingBoxRef.current && sceneRef.current) {
      sceneRef.current.remove(faceBoundingBoxRef.current);
      faceBoundingBoxRef.current = null;
    }
    
    if (faceNormalVectorRef.current && sceneRef.current) {
      sceneRef.current.remove(faceNormalVectorRef.current);
      faceNormalVectorRef.current = null;
    }
  }, []);

  const setupThreeJS = useCallback(() => {
    console.log('🔄 CALLBACK: setupThreeJS called');
    if (!canvasRef.current) return;
    if (sceneRef.current) return; // Already setup

    console.log(`🔧 Initializing Three.js scene...`);

    // Create scene
    sceneRef.current = new THREE.Scene();
    
    // Create renderer
    rendererRef.current = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: false,
    });
    
    // Set pixel ratio for crisp rendering on high-DPI displays
    rendererRef.current.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Create camera with default aspect ratio (will be updated when container is measured)
    cameraRef.current = new THREE.PerspectiveCamera(
      45,
      16 / 9, // Default aspect ratio
      0.1,
      1000
    );
    cameraRef.current.position.z = 2;
    
    // Apply device-based camera adjustments
    adjustCameraForDevice();

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

    console.log(`✅ Three.js scene initialized`);

    // Video texture will be created when video loads in createVideoTrack
  }, [adjustCameraForDevice]);

  // Create video track with device-appropriate resolution
  const createVideoTrack = useCallback(async () => {
    console.log('🔄 CALLBACK: createVideoTrack called');
    // Stop existing track if it exists
    if (videoTrackRef.current) {
      videoTrackRef.current.stop();
      videoTrackRef.current = null;
    }

    const resolution = getVideoResolution();

    try {
      const track = await createLocalVideoTrack({
        facingMode: "user",
        resolution: { 
          width: resolution.width, 
          height: resolution.height, 
          frameRate: 30 
        },
      });
      
      videoTrackRef.current = track;
      track.attach(videoRef.current!);
      
      // Set up event listeners for plane aspect updates
      const video = videoRef.current!;
      
      // Wait for video to be ready before starting
      video.addEventListener('loadedmetadata', () => {
        console.log(`Video loaded: ${video.videoWidth}x${video.videoHeight}`);
        console.log(`Video track settings:`, videoTrackRef.current?.mediaStreamTrack.getSettings());
        
        // Create video texture now that video is fully loaded
        if (video && sceneRef.current && !videoTextureRef.current) {
          videoTextureRef.current = new THREE.VideoTexture(video);
          videoTextureRef.current.flipY = true;
          videoTextureRef.current.colorSpace = THREE.SRGBColorSpace;
          videoTextureRef.current.minFilter = THREE.LinearFilter;
          videoTextureRef.current.magFilter = THREE.LinearFilter;
          videoTextureRef.current.format = THREE.RGBAFormat;
          videoTextureRef.current.generateMipmaps = false;
          videoTextureRef.current.wrapS = THREE.ClampToEdgeWrapping;
          videoTextureRef.current.wrapT = THREE.ClampToEdgeWrapping;
          
          console.log('Video texture created with consistent settings');
        }
        
        // Update plane aspect ratio based on actual video metadata
        updatePlaneAspect();
        
        // Start animation loop after video and texture are ready
        animate.current();
        
        // Setup FaceLandmarker after video is ready
        setTimeout(() => {
          setupFaceMesh();
        }, 1000);
      });
      
      // Add event listeners for aspect ratio updates
      video.addEventListener('loadedmetadata', updatePlaneAspect);
      window.addEventListener('resize', updatePlaneAspect);
      window.addEventListener('orientationchange', updatePlaneAspect);
      
      // Monitor for unwanted video track changes
      video.addEventListener('resize', () => {
        console.log(`Video resized: ${video.videoWidth}x${video.videoHeight}`);
        if (videoTrackRef.current) {
          console.log(`Track settings after resize:`, videoTrackRef.current.mediaStreamTrack.getSettings());
        }
        // Update plane aspect when video dimensions change
        updatePlaneAspect();
      });
    } catch (error) {
      console.error("Error creating video track:", error);
    }
  }, [setupFaceMesh, updatePlaneAspect, getVideoResolution]);

  useEffect(() => {  
    console.log('🎯 EFFECT [2/9]: Video track creation [createVideoTrack]');
    createVideoTrack();
    
    // Prevent video track recreation on orientation changes
    const handleOrientationChange = (e: Event) => {
      e.preventDefault();
      // Don't recreate video track on orientation changes
      console.log('Orientation change detected, maintaining existing video track');
    };
    
    // Listen for orientation changes but don't recreate video
    window.addEventListener('orientationchange', handleOrientationChange);
    
    return () => {
      window.removeEventListener('orientationchange', handleOrientationChange);
    };
  }, [createVideoTrack]);

  useEffect(() => {
    console.log('🎯 EFFECT [3/9]: Canvas resizing [size, size.height, size.width]', {size: size.width + 'x' + size.height});
    if (!canvasRef.current) return;
    if (!cameraRef.current) return;
    if (!rendererRef.current) return;
    
    // Use container dimensions if available, otherwise use canvas client dimensions
    const width = size.width || canvasRef.current.clientWidth || 800;
    const height = size.height || canvasRef.current.clientHeight || 600;
    
    console.log(`📐 Resizing canvas: ${width}x${height} (container: ${size.width}x${size.height})`);
    
    // Set canvas to fill the entire container
    canvasRef.current.width = width;
    canvasRef.current.height = height;
    
    rendererRef.current.setSize(width, height);
    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();
    
    console.log(`✅ Canvas resized to: ${width}x${height}`);
  }, [size, size.height, size.width]);

  useEffect(() => {
    console.log('🎯 EFFECT [4/9]: Canvas stream setup [onCanvasStreamChanged]');
    if (!canvasRef.current) return;
    if (canvasStreamRef.current) return;
    canvasStreamRef.current = canvasRef.current.captureStream(30);
    onCanvasStreamChanged(canvasStreamRef.current);
  }, [onCanvasStreamChanged]);

  // Initialize Three.js as soon as canvas is available
  useEffect(() => {
    console.log('🎯 EFFECT [5/9]: Early Three.js initialization [setupThreeJS]');
    if (canvasRef.current && !sceneRef.current) {
      console.log(`🎬 Canvas ready, initializing Three.js...`);
      setupThreeJS();
    }
  }, [setupThreeJS]);

  useEffect(() => {
    console.log('🎯 EFFECT [6/9]: Three.js setup [setupThreeJS]');
    setupThreeJS();
  }, [setupThreeJS]);

  // Adjust camera when device type changes
  useEffect(() => {
    console.log('🎯 EFFECT [7/9]: Camera adjustment [isMobile, adjustCameraForDevice]', {isMobile});
    adjustCameraForDevice();
  }, [isMobile, adjustCameraForDevice]);

  // Update plane aspect when device type changes (rare edge case)
  useEffect(() => {
    console.log('🎯 EFFECT [8/9]: Plane aspect update [isMobile, updatePlaneAspect]', {isMobile});
    if (sceneRef.current && videoTextureRef.current && videoRef.current) {
      updatePlaneAspect();
    }
  }, [isMobile, updatePlaneAspect]);

  // Cleanup video track and event listeners on unmount
  useEffect(() => {
    console.log('🎯 EFFECT [9/9]: Cleanup setup [updatePlaneAspect]');
    return () => {
      console.log('🧹 CLEANUP: Cleaning up video track and event listeners');
      if (videoTrackRef.current) {
        videoTrackRef.current.stop();
        videoTrackRef.current = null;
      }
      
      // Cleanup event listeners
      if (videoRef.current) {
        const video = videoRef.current;
        video.removeEventListener('loadedmetadata', updatePlaneAspect);
        video.removeEventListener('resize', updatePlaneAspect);
      }
      
    };
  }, [updatePlaneAspect]);

  return (
    <div className="relative h-full w-full bg-black" ref={resizeRef}>
      <canvas
        className="w-full h-full"
        ref={canvasRef}
      />
      <div className="absolute w-[0px] h-[0px] bottom-2 right-2 overflow-hidden">
        <video 
          className="h-full w-full" 
          ref={videoRef}
          style={{ objectFit: 'cover' }}
        />
      </div>
    </div>
  );
};
