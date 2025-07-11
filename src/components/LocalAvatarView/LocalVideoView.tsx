import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { createLocalVideoTrack } from "livekit-client";
import useResizeObserver from "use-resize-observer";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

type Props = {
  onCanvasStreamChanged: (canvasStream: MediaStream | null) => void;
};

export const LocalVideoView = ({ onCanvasStreamChanged }: Props) => {
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
  const faceBoundingBoxRef = useRef<THREE.LineSegments | null>(null);
  const faceNormalVectorRef = useRef<THREE.ArrowHelper | null>(null);
  const [showFaceBoundingBox, setShowFaceBoundingBox] = useState(false);
  const [videoAspectRatio, setVideoAspectRatio] = useState<number>(9/16);
  const size = useResizeObserver({ ref: resizeRef });

  const updateVideoPlaneSize = useCallback(() => {
    if (!videoRef.current || !planeRef.current || !cameraRef.current) return;
    
    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;
    
    // 1. Get the aspect ratio of the webcam being captured
    const actualAspectRatio = video.videoWidth / video.videoHeight;
    setVideoAspectRatio(actualAspectRatio);
    
    console.log(`Video dimensions: ${video.videoWidth}x${video.videoHeight}, aspect ratio: ${actualAspectRatio}`);
    
    // 2. Create a video texture plane that matches the aspect ratio of the capture video
    const planeWidth = 2; // Fixed width
    const planeHeight = planeWidth / actualAspectRatio; // Height calculated from video aspect ratio
    
    // Create new geometry with correct aspect ratio
    const newGeometry = new THREE.PlaneGeometry(planeWidth, planeHeight, 128, 192);
    planeRef.current.geometry.dispose(); // Clean up old geometry
    planeRef.current.geometry = newGeometry;
    
    // 3. Maintain a 9:16 frustum for the threejs camera (fixed dimensions)
    const cameraAspectRatio = 9 / 16; // Fixed 9:16 aspect ratio for camera
    const frustumWidth = 2; // Fixed frustum width
    const frustumHeight = frustumWidth / cameraAspectRatio; // Fixed frustum height for 9:16
    
    cameraRef.current.left = -frustumWidth / 2;
    cameraRef.current.right = frustumWidth / 2;
    cameraRef.current.top = frustumHeight / 2;
    cameraRef.current.bottom = -frustumHeight / 2;
    cameraRef.current.updateProjectionMatrix();
    
    // 4. Adjust the positioning of the camera to fit the width of the video plane
    // Calculate the scale factor to fit the video plane width within the camera frustum
    const scaleFactor = frustumWidth / planeWidth;
    
    // Adjust camera position Z to achieve the desired scale
    // Moving camera closer (smaller Z) makes objects appear larger
    // Moving camera farther (larger Z) makes objects appear smaller
    const baseDistance = 5; // Base camera distance
    const adjustedDistance = baseDistance / scaleFactor;
    
    cameraRef.current.position.set(0, 0, adjustedDistance);
    
    console.log(`Updated plane size: ${planeWidth}x${planeHeight}, camera frustum: ${frustumWidth}x${frustumHeight} (9:16), camera distance: ${adjustedDistance}`);
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

  const setupFaceLandmarker = useCallback(async () => {
    // Ensure we're running on client side
    if (typeof window === 'undefined') return;
    
    try {
      console.log("Initializing FaceLandmarker for face detection...");
      
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
      const detectFaceLandmarks = () => {
        if (faceLandmarkerRef.current && videoRef.current && videoRef.current.videoWidth > 0) {
          try {
            const startTimeMs = performance.now();
            const results = faceLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
            
            if (results.faceLandmarks && results.faceLandmarks.length > 0) {
              createOrUpdateFaceBoundingBox(results.faceLandmarks);
            }
           
          } catch (detectionError) {
            console.warn("Face landmark detection error:", detectionError);
          }
        }
        requestAnimationFrame(detectFaceLandmarks);
      };
      
      // Wait for video to be fully ready
      setTimeout(() => {
        detectFaceLandmarks();
      }, 500);
      
    } catch (error) {
      console.error("Error setting up FaceLandmarker:", error);
      // Retry after a delay
      setTimeout(() => {
        console.log("Retrying FaceLandmarker setup...");
        setupFaceLandmarker();
      }, 2000);
    }
  }, []);

  const setupThreeJS = useCallback(() => {
    if (!canvasRef.current) return;
    if (sceneRef.current) return; // Already setup
    if (!size.width || !size.height) return;

    // Calculate canvas dimensions based on 9:16 camera frustum (not video aspect ratio)
    const cameraAspectRatio = 9 / 16; // Fixed 9:16 aspect ratio for camera
    const canvasHeight = size.height;
    const canvasWidth = canvasHeight * cameraAspectRatio; // Use camera aspect ratio for canvas

    // Create scene
    sceneRef.current = new THREE.Scene();
    
    // Create renderer with better quality settings
    rendererRef.current = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: false,
      premultipliedAlpha: false,
    });
    rendererRef.current.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    rendererRef.current.setSize(canvasWidth, canvasHeight, false);
    
    // Create orthographic camera with fixed 9:16 frustum
    // 2. Create a video texture plane that matches the aspect ratio of the capture video
    const planeWidth = 2; // Fixed width
    const planeHeight = planeWidth / videoAspectRatio; // Height calculated from video aspect ratio
    
    // 3. Maintain a 9:16 frustum for the threejs camera (fixed dimensions)
    const frustumWidth = 2; // Fixed frustum width
    const frustumHeight = frustumWidth / cameraAspectRatio; // Fixed frustum height for 9:16
    
    cameraRef.current = new THREE.OrthographicCamera(
      -frustumWidth / 2,   // left
      frustumWidth / 2,    // right
      frustumHeight / 2,   // top
      -frustumHeight / 2,  // bottom
      0.1,                 // near
      1000                 // far
    );
    
    // 4. Adjust the positioning of the camera to fit the width of the video plane
    // Calculate the scale factor to fit the video plane width within the camera frustum
    const scaleFactor = frustumWidth / planeWidth;
    const baseDistance = 5; // Base camera distance
    const adjustedDistance = baseDistance / scaleFactor;
    
    cameraRef.current.position.set(0, 0, adjustedDistance);

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
      videoTextureRef.current.format = THREE.RGBAFormat;
      videoTextureRef.current.generateMipmaps = false;

      // Create basic material for video display
      const videoMaterial = new THREE.MeshBasicMaterial({
        map: videoTextureRef.current
      });

      // Create plane geometry with dynamic aspect ratio
      const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight, 128, 192);
      
      planeRef.current = new THREE.Mesh(geometry, videoMaterial);
      sceneRef.current.add(planeRef.current);

      // Add event listeners to update plane size when video metadata loads
      const video = videoRef.current;
      const handleLoadedMetadata = () => {
        console.log('Video metadata loaded, updating plane size');
        setTimeout(() => updateVideoPlaneSize(), 100); // Small delay to ensure dimensions are available
      };
      
      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      video.addEventListener('resize', updateVideoPlaneSize);
      
      // Cleanup function
      return () => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        video.removeEventListener('resize', updateVideoPlaneSize);
      };
    }
  }, [size.height, size.width, videoAspectRatio, updateVideoPlaneSize]);

  const toggleFaceBoundingBox = useCallback(() => {
    setShowFaceBoundingBox(prev => {
      const newValue = !prev;
      console.log('Face bounding box toggle:', prev, '->', newValue);
      if (!newValue) {
        removeFaceBoundingBox();
      }
      return newValue;
    });
  }, []);

  // Expose control functions globally for testing
  useEffect(() => {
    if (typeof window !== 'undefined') {
      (window as any).faceControls = {
        toggleFaceBoundingBox,
        getFaceBoundingBoxVisible: () => showFaceBoundingBox
      };
      
      console.log("Face detection controls available:");
      console.log("window.faceControls.toggleFaceBoundingBox() // toggle face bounding box visibility");
    }
  }, [toggleFaceBoundingBox, showFaceBoundingBox]);

  useEffect(() => {  
    createLocalVideoTrack({
      facingMode: "user",
      resolution: { 
        width: 1080, 
        height: 1920, 
        frameRate: 60 
      },
    }).then((t) => {
      t.attach(videoRef.current!);
      // Start animation loop after video is attached
      animate.current();
      
      // Setup FaceLandmarker after video is ready
      setTimeout(() => {
        setupFaceLandmarker();
      }, 2000);
    }).catch((error) => {
      console.error("Error creating video track with preferred resolution, trying fallback:", error);
      // Fallback to other 9:16 resolutions if the preferred one fails
      const fallbackResolutions = [
        { width: 720, height: 1280 }, // 720p 9:16
        { width: 540, height: 960 },  // 540p 9:16
        { width: 360, height: 640 },  // 360p 9:16
      ];
      
      const tryFallback = async (resolutions: typeof fallbackResolutions) => {
        for (const resolution of resolutions) {
          try {
            const track = await createLocalVideoTrack({
              facingMode: "user",
              resolution: { 
                ...resolution,
                frameRate: 60 
              },
            });
            track.attach(videoRef.current!);
            animate.current();
            console.log(`Successfully created video track with resolution: ${resolution.width}x${resolution.height}`);
            
            // Setup FaceLandmarker after video is ready
            setTimeout(() => {
              setupFaceLandmarker();
            }, 2000);
            return;
          } catch (fallbackError) {
            console.warn(`Failed to create video track with resolution ${resolution.width}x${resolution.height}:`, fallbackError);
          }
        }
        throw new Error("All video track creation attempts failed");
      };
      
      tryFallback(fallbackResolutions).catch((finalError) => {
        console.error("Failed to create video track with any resolution:", finalError);
      });
    });
  }, [setupFaceLandmarker]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (!cameraRef.current) return;
    if (!size.width || !size.height) return;
    
    // Calculate canvas dimensions based on 9:16 camera frustum (not video aspect ratio)
    const cameraAspectRatio = 9 / 16; // Fixed 9:16 aspect ratio for camera
    const canvasHeight = size.height;
    const canvasWidth = canvasHeight * cameraAspectRatio; // Use camera aspect ratio for canvas
    
    // Set canvas dimensions with proper pixel ratio handling
    const pixelRatio = Math.min(window.devicePixelRatio, 2);
    canvasRef.current.width = canvasWidth * pixelRatio;
    canvasRef.current.height = canvasHeight * pixelRatio;
    canvasRef.current.style.width = `${canvasWidth}px`;
    canvasRef.current.style.height = `${canvasHeight}px`;
    rendererRef.current?.setSize(canvasWidth, canvasHeight, false);
    
    // Update orthographic camera frustum with fixed 9:16 dimensions
    // 2. Create a video texture plane that matches the aspect ratio of the capture video
    const planeWidth = 2; // Fixed width
    const planeHeight = planeWidth / videoAspectRatio; // Height calculated from video aspect ratio
    
    // 3. Maintain a 9:16 frustum for the threejs camera (fixed dimensions)
    const frustumWidth = 2; // Fixed frustum width
    const frustumHeight = frustumWidth / cameraAspectRatio; // Fixed frustum height for 9:16
    
    cameraRef.current.left = -frustumWidth / 2;
    cameraRef.current.right = frustumWidth / 2;
    cameraRef.current.top = frustumHeight / 2;
    cameraRef.current.bottom = -frustumHeight / 2;
    cameraRef.current.updateProjectionMatrix();

    // 4. Adjust the positioning of the camera to fit the width of the video plane
    // Calculate the scale factor to fit the video plane width within the camera frustum
    const scaleFactor = frustumWidth / planeWidth;
    const baseDistance = 5; // Base camera distance
    const adjustedDistance = baseDistance / scaleFactor;
    
    cameraRef.current.position.set(0, 0, adjustedDistance);
  }, [size, size.height, size.width, videoAspectRatio]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (canvasStreamRef.current) return;
    canvasStreamRef.current = canvasRef.current.captureStream(60);
    onCanvasStreamChanged(canvasStreamRef.current);
  }, [onCanvasStreamChanged]);

  useEffect(setupThreeJS, [setupThreeJS]);

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

  return (
    <div className="relative h-full w-full">
      <div className="overflow-hidden h-full flex items-center justify-center" ref={resizeRef}>
        <canvas
          className="h-full"
          style={{ aspectRatio: '9/16' }}
          ref={canvasRef}
        />
      </div>
      <div className="absolute w-[0px] h-[0px] bottom-2 right-2 overflow-hidden">
        <video className="h-full w-full" ref={videoRef} />
      </div>
    </div>
  );
};
