import { useCallback, useEffect, useRef, useState } from "react";
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
  const faceBoundingBoxRef = useRef<THREE.LineSegments | null>(null);
  const faceNormalVectorRef = useRef<THREE.ArrowHelper | null>(null);
  const [showFaceBoundingBox, setShowFaceBoundingBox] = useState(false);
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
  }, [updateLipDeformation]);

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

      // Create plane geometry with high subdivision for smooth deformation
      const geometry = new THREE.PlaneGeometry(2, 1.5, 128, 96);
      
      planeRef.current = new THREE.Mesh(geometry, lipShaderRef.current);
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
      (window as any).lipControls = {
        setSmileIntensity,
        setGrimaceIntensity,
        setDeformationRadius,
        setAnchorOffsetX,
        setAnchorOffsetY,
        toggleFaceBoundingBox,
        getCurrentIntensity: () => deformationIntensity.current,
        getCurrentRadius: () => deformationRadius.current,
        getCurrentAnchorOffset: () => ({ x: anchorOffsetX.current, y: anchorOffsetY.current }),
        getFaceBoundingBoxVisible: () => showFaceBoundingBox
      };
      
      console.log("Lip deformation controls available:");
      console.log("window.lipControls.setSmileIntensity(1.0) // 0.0 to 3.0");
      console.log("window.lipControls.setGrimaceIntensity(1.0) // 0.0 to 3.0");
      console.log("window.lipControls.setDeformationRadius(0.15) // 0.05 to 0.3");
      console.log("window.lipControls.setAnchorOffsetX(0.0) // -0.2 to 0.2");
      console.log("window.lipControls.setAnchorOffsetY(0.02) // -0.2 to 0.2");
      console.log("window.lipControls.toggleFaceBoundingBox() // toggle face bounding box visibility");
    }
  }, [setSmileIntensity, setGrimaceIntensity, setDeformationRadius, setAnchorOffsetX, setAnchorOffsetY, toggleFaceBoundingBox, showFaceBoundingBox]);

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
      <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white p-4 rounded max-w-xs">
        <h3 className="text-sm font-bold mb-2">Lip Deformation Controls</h3>
        
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
          <button 
            onClick={toggleFaceBoundingBox}
            className={`block w-full px-2 py-1 rounded ${
              showFaceBoundingBox 
                ? 'bg-cyan-600 hover:bg-cyan-700' 
                : 'bg-gray-600 hover:bg-gray-700'
            }`}
          >
            {showFaceBoundingBox ? 'Hide Face Box' : 'Show Face Box'}
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
