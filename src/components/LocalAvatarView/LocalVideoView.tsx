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

// Lip Deformation Shader
const lipDeformationVertexShader = `
  varying vec2 vUv;
  uniform vec2 lipCenter;
  uniform vec2 anchorOffset;
  uniform float deformationIntensity;
  uniform float deformationRadius;
  
  void main() {
    vUv = uv;
    vec3 pos = position;
    
    // Apply anchor point displacement to lip center
    vec2 adjustedLipCenter = lipCenter + anchorOffset;
    
    // Calculate distance from current vertex to adjusted lip center
    float distanceToLip = distance(uv, adjustedLipCenter);
    
    // Apply radial deformation under the lips
    if (distanceToLip < deformationRadius) {
      // Calculate falloff factor (stronger at center, weaker at edges)
      float factor = (deformationRadius - distanceToLip) / deformationRadius;
      factor = smoothstep(0.0, 1.0, factor); // Smooth falloff
      
      // Calculate direction from adjusted lip center to current point
      vec2 direction = normalize(uv - adjustedLipCenter);
      
      // Apply deformation - FIXED: Inverted Y direction
      // Positive values create smile (upward), negative create grimace (downward)
      pos.y -= factor * deformationIntensity * 0.1; // Flipped sign for correct direction
      
      // Add slight horizontal spread for more natural look
      pos.x += direction.x * factor * deformationIntensity * 0.03;
    }
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const lipDeformationFragmentShader = `
  uniform sampler2D videoTexture;
  varying vec2 vUv;
  
  void main() {
    vec4 videoColor = texture2D(videoTexture, vUv);
    gl_FragColor = videoColor;
  }
`;

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
  const faceMeshRef = useRef<THREE.LineSegments | null>(null);
  const faceGeometryRef = useRef<THREE.BufferGeometry | null>(null);
  const faceMaterialRef = useRef<THREE.LineBasicMaterial | null>(null);
  const lipShaderRef = useRef<THREE.ShaderMaterial | null>(null);
  const size = useResizeObserver({ ref: resizeRef });

  // Official MediaPipe face mesh indices for specific facial features
  const faceIndices = useRef<number[]>([]);

  // Deformation controls
  const deformationIntensity = useRef(0.0); // Start with filters OFF - Positive = smile, Negative = grimace
  const deformationRadius = useRef(0.15); // Size of affected area
  const anchorOffsetX = useRef(0.0); // Horizontal anchor displacement
  const anchorOffsetY = useRef(0.02); // Vertical anchor displacement (default slightly below lip)

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

  const updateLipDeformation = useCallback((faceLandmarks: any[]) => {
    if (!lipShaderRef.current || !faceLandmarks || faceLandmarks.length === 0) return;
    
    const landmarks = faceLandmarks[0];
    
    // Use lip center landmark (MediaPipe landmark 13 is mouth center)
    const lipCenter = landmarks[13];
    
    // Convert landmark to UV coordinates - FIXED: Coordinate mapping
    // MediaPipe coordinates are normalized [0,1] where (0,0) is top-left
    // UV coordinates are [0,1] where (0,0) is bottom-left
    const lipUV = new THREE.Vector2(lipCenter.x, 1.0 - lipCenter.y);
    
    // Update shader uniforms
    lipShaderRef.current.uniforms.lipCenter.value = lipUV;
    lipShaderRef.current.uniforms.anchorOffset.value = new THREE.Vector2(anchorOffsetX.current, anchorOffsetY.current);
    lipShaderRef.current.uniforms.deformationIntensity.value = deformationIntensity.current;
    lipShaderRef.current.uniforms.deformationRadius.value = deformationRadius.current;
  }, []);

  const setupFaceLandmarker = useCallback(async () => {
    // Ensure we're running on client side
    if (typeof window === 'undefined') return;
    
    try {
      console.log("Initializing FaceLandmarker for lip deformation...");
      
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
              updateLipDeformation(results.faceLandmarks);
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
  }, [updateLipDeformation]);

  const setupThreeJS = useCallback(() => {
    if (!canvasRef.current) return;
    if (sceneRef.current) return; // Already setup
    if (!size.width || !size.height) return;

    // Create scene
    sceneRef.current = new THREE.Scene();
    
    // Create renderer with higher pixel ratio for better quality
    rendererRef.current = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance"
    });
    
    // Set pixel ratio for crisp rendering
    rendererRef.current.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Calculate iPhone 15 aspect ratio
    const videoAspectRatio = 1170 / 2532; // ~0.462
    
    // Create orthographic camera positioned head-on to the video texture
    // Use video aspect ratio to prevent stretching
    const frustumHeight = 1.0; // Controls zoom level
    const frustumWidth = frustumHeight * videoAspectRatio; // Maintain video aspect ratio
    
    cameraRef.current = new THREE.OrthographicCamera(
      -frustumWidth / 2,   // left
      frustumWidth / 2,    // right
      frustumHeight / 2,   // top
      -frustumHeight / 2,  // bottom
      0.1,                 // near
      1000                 // far
    );
    
    // Position camera head-on to the video texture (straight down Z-axis)
    cameraRef.current.position.set(0, 0, 2);
    cameraRef.current.lookAt(0, 0, 0);

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
      
      // Use higher quality filtering for better resolution
      videoTextureRef.current.minFilter = THREE.LinearFilter;
      videoTextureRef.current.magFilter = THREE.LinearFilter;
      videoTextureRef.current.generateMipmaps = false; // Disable mipmaps for video textures
      videoTextureRef.current.wrapS = THREE.ClampToEdgeWrapping;
      videoTextureRef.current.wrapT = THREE.ClampToEdgeWrapping;

      // Create lip deformation shader material
      lipShaderRef.current = new THREE.ShaderMaterial({
        uniforms: {
          videoTexture: { value: videoTextureRef.current },
          lipCenter: { value: new THREE.Vector2(0.5, 0.7) }, // Default position
          anchorOffset: { value: new THREE.Vector2(anchorOffsetX.current, anchorOffsetY.current) },
          deformationIntensity: { value: deformationIntensity.current },
          deformationRadius: { value: deformationRadius.current }
        },
        vertexShader: lipDeformationVertexShader,
        fragmentShader: lipDeformationFragmentShader
      });

      // Create plane geometry that exactly matches the video aspect ratio
      // This prevents any stretching by matching the orthographic frustum
      const planeWidth = frustumWidth;   // Match orthographic frustum width exactly
      const planeHeight = frustumHeight; // Match orthographic frustum height exactly
      
      const geometry = new THREE.PlaneGeometry(
        planeWidth, 
        planeHeight, 
        512, // High subdivision for smooth deformation
        Math.floor(512 / videoAspectRatio) // Proportional subdivision based on aspect ratio
      );
      
      planeRef.current = new THREE.Mesh(geometry, lipShaderRef.current);
      
      // Position plane to fill the orthographic view exactly
      planeRef.current.position.set(0, 0, 0);
      
      sceneRef.current.add(planeRef.current);
    }
  }, [size.height, size.width]);

  // Control functions for deformation
  const setSmileIntensity = useCallback((intensity: number) => {
    deformationIntensity.current = Math.abs(intensity); // Positive for smile
    if (lipShaderRef.current) {
      lipShaderRef.current.uniforms.deformationIntensity.value = deformationIntensity.current;
    }
  }, []);

  const setGrimaceIntensity = useCallback((intensity: number) => {
    deformationIntensity.current = -Math.abs(intensity); // Negative for grimace
    if (lipShaderRef.current) {
      lipShaderRef.current.uniforms.deformationIntensity.value = deformationIntensity.current;
    }
  }, []);

  const setDeformationRadius = useCallback((radius: number) => {
    deformationRadius.current = Math.max(0.05, Math.min(0.3, radius)); // Clamp between 0.05 and 0.3
    if (lipShaderRef.current) {
      lipShaderRef.current.uniforms.deformationRadius.value = deformationRadius.current;
    }
  }, []);

  const setAnchorOffsetX = useCallback((offsetX: number) => {
    anchorOffsetX.current = Math.max(-0.2, Math.min(0.2, offsetX)); // Clamp between -0.2 and 0.2
    if (lipShaderRef.current) {
      lipShaderRef.current.uniforms.anchorOffset.value = new THREE.Vector2(anchorOffsetX.current, anchorOffsetY.current);
    }
  }, []);

  const setAnchorOffsetY = useCallback((offsetY: number) => {
    anchorOffsetY.current = Math.max(-0.2, Math.min(0.2, offsetY)); // Clamp between -0.2 and 0.2
    if (lipShaderRef.current) {
      lipShaderRef.current.uniforms.anchorOffset.value = new THREE.Vector2(anchorOffsetX.current, anchorOffsetY.current);
    }
  }, []);

  // Expose control functions globally for testing
  useEffect(() => {
    if (typeof window !== 'undefined') {
      (window as any).lipControls = {
        setSmileIntensity,
        setGrimaceIntensity,
        setDeformationRadius,
        setAnchorOffsetX,
        setAnchorOffsetY,
        getCurrentIntensity: () => deformationIntensity.current,
        getCurrentRadius: () => deformationRadius.current,
        getCurrentAnchorOffset: () => ({ x: anchorOffsetX.current, y: anchorOffsetY.current })
      };
      
      console.log("Lip deformation controls available:");
      console.log("window.lipControls.setSmileIntensity(1.0) // 0.0 to 3.0");
      console.log("window.lipControls.setGrimaceIntensity(1.0) // 0.0 to 3.0");
      console.log("window.lipControls.setDeformationRadius(0.15) // 0.05 to 0.3");
      console.log("window.lipControls.setAnchorOffsetX(0.0) // -0.2 to 0.2");
      console.log("window.lipControls.setAnchorOffsetY(0.02) // -0.2 to 0.2");
    }
  }, [setSmileIntensity, setGrimaceIntensity, setDeformationRadius, setAnchorOffsetX, setAnchorOffsetY]);

  useEffect(() => {  
    createLocalVideoTrack({
      facingMode: "user",
      resolution: { 
        width: 1170,   // iPhone 15 vertical resolution width
        height: 2532,  // iPhone 15 vertical resolution height 
        frameRate: 30 
      }
    }).then((t) => {
      t.attach(videoRef.current!);
      // Start animation loop after video is attached
      animate.current();
      
      // Setup FaceLandmarker after video is ready
      setTimeout(() => {
        setupFaceLandmarker();
      }, 2000);
    });
  }, [setupFaceLandmarker]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (!cameraRef.current) return;
    if (!size.width || !size.height) return;
    
    // Calculate iPhone 15 aspect ratio (9:19.5 approximately)
    const videoAspectRatio = 1170 / 2532; // ~0.462
    const containerAspectRatio = size.width / size.height;
    
    let canvasWidth, canvasHeight;
    
    // Determine if we're on mobile (screen width < 768px)
    const isMobile = window.innerWidth < 768;
    
    if (isMobile) {
      // On mobile: Use full screen iPhone 15 dimensions, centered
      canvasWidth = Math.min(size.width, window.innerWidth);
      canvasHeight = canvasWidth / videoAspectRatio;
      
      // If height exceeds container, scale down
      if (canvasHeight > size.height) {
        canvasHeight = size.height;
        canvasWidth = canvasHeight * videoAspectRatio;
      }
    } else {
      // On desktop: Maintain video aspect ratio but fit within container
      if (containerAspectRatio > videoAspectRatio) {
        // Container is wider than video aspect ratio
        canvasHeight = size.height;
        canvasWidth = canvasHeight * videoAspectRatio;
      } else {
        // Container is taller than video aspect ratio
        canvasWidth = size.width;
        canvasHeight = canvasWidth / videoAspectRatio;
      }
    }
    
    canvasRef.current.width = canvasWidth;
    canvasRef.current.height = canvasHeight;
    rendererRef.current?.setSize(canvasWidth, canvasHeight);
    
    // Update orthographic camera frustum - maintain video aspect ratio to prevent stretching
    const frustumHeight = 1.0; // Match the setup value
    const frustumWidth = frustumHeight * videoAspectRatio; // Always use video aspect ratio
    
    cameraRef.current.left = -frustumWidth / 2;
    cameraRef.current.right = frustumWidth / 2;
    cameraRef.current.top = frustumHeight / 2;
    cameraRef.current.bottom = -frustumHeight / 2;
    cameraRef.current.updateProjectionMatrix();
  }, [size, size.height, size.width]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (canvasStreamRef.current) return;
    
    // Capture stream at higher frame rate for better quality
    canvasStreamRef.current = canvasRef.current.captureStream(60);
    onCanvasStreamChanged(canvasStreamRef.current);
  }, [onCanvasStreamChanged]);

  useEffect(setupThreeJS, [setupThreeJS]);

  return (
    <div className="relative h-full w-full flex items-center justify-center bg-black">
      <div className="relative" ref={resizeRef} style={{ width: '100%', height: '100%', maxWidth: '100vw', maxHeight: '100vh' }}>
        <canvas
          className="block mx-auto"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            maxWidth: '100vw',
            maxHeight: '100vh'
          }}
          ref={canvasRef}
        />
      </div>
      <div className="absolute w-[0px] h-[0px] bottom-2 right-2 overflow-hidden">
        <video className="h-full w-full" ref={videoRef} />
      </div>
      
      {/* Control Panel */}
      <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white p-4 rounded max-w-xs z-10">
        <h3 className="text-sm font-bold mb-2">Lip Deformation Controls</h3>
        <p className="text-xs text-gray-300 mb-3">iPhone 15 Resolution • 1170×2532</p>
        
        {/* Intensity Controls */}
        <div className="space-y-2 text-xs mb-4">
          <button 
            onClick={() => setSmileIntensity(1.5)}
            className="block w-full bg-green-600 hover:bg-green-700 px-2 py-1 rounded"
          >
            Smile
          </button>
          <button 
            onClick={() => setGrimaceIntensity(1.5)}
            className="block w-full bg-red-600 hover:bg-red-700 px-2 py-1 rounded"
          >
            Grimace
          </button>
          <button 
            onClick={() => setSmileIntensity(0)}
            className="block w-full bg-gray-600 hover:bg-gray-700 px-2 py-1 rounded"
          >
            Reset
          </button>
        </div>

        {/* Radial Distance Control */}
        <div className="mb-3">
          <label className="text-xs block mb-1">Radial Distance: {deformationRadius.current.toFixed(2)}</label>
          <input
            type="range"
            min="0.05"
            max="0.3"
            step="0.01"
            defaultValue={deformationRadius.current}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            onChange={(e) => setDeformationRadius(parseFloat(e.target.value))}
          />
        </div>

        {/* Anchor Point Controls */}
        <div className="mb-3">
          <label className="text-xs block mb-1">Anchor Offset X: {anchorOffsetX.current.toFixed(2)}</label>
          <input
            type="range"
            min="-0.2"
            max="0.2"
            step="0.01"
            defaultValue={anchorOffsetX.current}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            onChange={(e) => setAnchorOffsetX(parseFloat(e.target.value))}
          />
        </div>

        <div className="mb-3">
          <label className="text-xs block mb-1">Anchor Offset Y: {anchorOffsetY.current.toFixed(2)}</label>
          <input
            type="range"
            min="-0.2"
            max="0.2"
            step="0.01"
            defaultValue={anchorOffsetY.current}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            onChange={(e) => setAnchorOffsetY(parseFloat(e.target.value))}
          />
        </div>
      </div>
    </div>
  );
};
