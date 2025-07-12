export const StaticShader = {
  uniforms: {
    tDiffuse: { value: null },
    time: { value: 0.0 },
    amount: { value: 0.5 },
    size: { value: 4.0 },
  },

  vertexShader: `
    varying vec2 vUv;

    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
    }
  `,

  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float time;
    uniform float amount;
    uniform float size;

    varying vec2 vUv;

    float rand(vec2 co){
      return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
    }

    void main() {
      vec2 p = vUv;
      vec4 color = texture2D(tDiffuse, p);
      float xs = floor(gl_FragCoord.x / size);
      float ys = floor(gl_FragCoord.y / size);
      vec4 snow = vec4(rand(vec2(xs * time, ys * time)) * amount);

      // Additive blending
      gl_FragColor = color + snow;
    }
  `,
};
