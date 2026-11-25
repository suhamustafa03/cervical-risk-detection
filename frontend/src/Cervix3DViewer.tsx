
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

type ViewMode = "front" | "side";
type DisplayMode = "surface" | "cutaway";

export default function Cervix3DViewer() {
  const mountRef = useRef<HTMLDivElement>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const cutPlaneRef = useRef<THREE.Plane | null>(null);
  const depthGuideRef = useRef<THREE.Group | null>(null);
  const showDepthGuideRef = useRef(true);
  

  const [view, setView] = useState<ViewMode>("front");
  const [mode, setMode] = useState<DisplayMode>("surface");
  const [cutPosition, setCutPosition] = useState(0);
  const [showDepthGuide, setShowDepthGuide] = useState(true);
  const [status, setStatus] = useState("Loading anatomical model...");

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let disposed = false;
    let frame = 0;
    let loadedModel: THREE.Object3D | null = null;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f0fafb");

    const camera = new THREE.PerspectiveCamera(
      40,
      mount.clientWidth / Math.max(mount.clientHeight, 1),
      0.01,
      100
    );
    camera.position.set(0, 0.2, 4.5);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.localClippingEnabled = true;
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x8c6970, 2));

    const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
    keyLight.position.set(3, 4, 5);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 1.2);
    fillLight.position.set(-4, 1, -3);
    scene.add(fillLight);

    const anatomy = new THREE.Group();
    scene.add(anatomy);

    const cutPlane = new THREE.Plane(
      new THREE.Vector3(1, 0, 0),
      0
    );
    cutPlaneRef.current = cutPlane;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1.8;
    controls.maxDistance = 9;
    controlsRef.current = controls;

    const loader = new GLTFLoader();

    loader.load(
      "/models/reproductive_anatomy.glb",
      (gltf) => {
        if (disposed) {
          gltf.scene.traverse((object) => {
            if (object instanceof THREE.Mesh) {
              object.geometry.dispose();
              const materials = Array.isArray(object.material)
                ? object.material
                : [object.material];
              materials.forEach((material) => material.dispose());
            }
          });
          return;
        }

        loadedModel = gltf.scene;

        const bounds = new THREE.Box3().setFromObject(loadedModel);
        const size = bounds.getSize(new THREE.Vector3());
        const center = bounds.getCenter(new THREE.Vector3());
        const largestDimension = Math.max(size.x, size.y, size.z);

        if (!Number.isFinite(largestDimension) || largestDimension <= 0) {
          setStatus("The model has no usable 3D geometry.");
          return;
        }

        // Centre and normalise the model for the viewer.
        const scale = 2.6 / largestDimension;
        loadedModel.scale.setScalar(scale);
        loadedModel.position.copy(center).multiplyScalar(-scale);

        // Preserve original textures and materials.
        // Enable clipping only on the model's cloned materials.
        loadedModel.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return;

          object.material = Array.isArray(object.material)
            ? object.material.map((material) => {
                const clone = material.clone();
                clone.clippingPlanes = [cutPlane];
                clone.clipShadows = true;
                clone.needsUpdate = true;
                return clone;
              })
            : (() => {
                const clone = object.material.clone();
                clone.clippingPlanes = [cutPlane];
                clone.clipShadows = true;
                clone.needsUpdate = true;
                return clone;
              })();

          object.castShadow = true;
          object.receiveShadow = true;
        });

        anatomy.add(loadedModel);

      // Illustrative tissue-depth guide (not a lesion prediction).
      const depthGuide = new THREE.Group();

      const guideMaterial = new THREE.MeshBasicMaterial({
        color: "#168BFF",
        transparent: true,
        opacity: 0.8,
        depthTest: false,
      });

      // A thin reference line and a small marker.
      const guideLine = new THREE.Mesh(
        new THREE.CylinderGeometry(0.012, 0.012, 0.7, 12),
        guideMaterial
      );
      guideLine.position.set(0.05, -0.45, 0.22);
      depthGuide.add(guideLine);

      const guideMarker = new THREE.Mesh(
        new THREE.SphereGeometry(0.045, 16, 12),
        guideMaterial
      );
      guideMarker.position.set(0.05, -0.8, 0.22);
      depthGuide.add(guideMarker);

      depthGuide.visible = showDepthGuideRef.current;
      anatomy.add(depthGuide);
      depthGuideRef.current = depthGuide;


        
        // The centre of the normalised model is at the origin.
        controls.target.set(0, 0, 0);
        camera.position.set(0, 0.15, 4.5);
        camera.lookAt(0, 0, 0);
        controls.update();

        setStatus("Anatomical model loaded.");
      },
      undefined,
      (error) => {
        console.error("Could not load the anatomical GLB:", error);
        if (!disposed) {
          setStatus(
            "Could not load the model. Check the file path and browser console."
          );
        }
      }
    );

    const render = () => {
      frame = requestAnimationFrame(render);
      controls.update();
      renderer.render(scene, camera);
    };
    render();

    const resize = () => {
      if (!mount.clientWidth || !mount.clientHeight) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();

      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material)
            ? object.material
            : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });

      renderer.domElement.remove();
      controlsRef.current = null;
      cameraRef.current = null;
      cutPlaneRef.current = null;
    
    };
  }, []);

  useEffect(() => {
    const plane = cutPlaneRef.current;
    if (!plane) return;

    // Move the clipping plane across the model.
    plane.constant = mode === "cutaway" ? cutPosition : 0;
  }, [cutPosition, mode]);

  useEffect(() => {
    showDepthGuideRef.current = showDepthGuide;
    if (depthGuideRef.current) {
      depthGuideRef.current.visible = showDepthGuide;
    }
  }, [showDepthGuide]);

  

  const changeView = (nextView: ViewMode) => {
    setView(nextView);

    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;

    camera.position.set(
      nextView === "front" ? 0 : 4.5,
      0.15,
      nextView === "front" ? 4.5 : 0
    );
    controls.target.set(0, 0, 0);
    camera.lookAt(0, 0, 0);
    controls.update();
  };

  return (
    <section className="results-image-card">
      <h3>Interactive 3D anatomy</h3>
      <p>
        Explore the downloaded anatomical model. Drag to rotate and
        use the scroll wheel or trackpad to zoom.
      </p>

      <div className="heatmap-toggle-row">
        <button
          type="button"
          className={view === "front" ? "primary-button" : "secondary-button"}
          onClick={() => changeView("front")}
        >
          Front view
        </button>
        <button
          type="button"
          className={view === "side" ? "primary-button" : "secondary-button"}
          onClick={() => changeView("side")}
        >
          Side view
        </button>
      </div>

      <div className="heatmap-toggle-row">
        <button
          type="button"
          className={mode === "surface" ? "primary-button" : "secondary-button"}
          onClick={() => setMode("surface")}
        >
          Surface view
        </button>
        <button
          type="button"
          className={mode === "cutaway" ? "primary-button" : "secondary-button"}
          onClick={() => setMode("cutaway")}
        >
          Interactive cutaway
        </button>
      </div>
      <label
  style={{
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginTop: 12,
  }}
>
  <input
    type="checkbox"
    checked={showDepthGuide}
    onChange={(event) => setShowDepthGuide(event.target.checked)}
  />
  Show illustrative tissue-depth guide
</label>
      {mode === "cutaway" && (
        <div style={{ margin: "12px 0" }}>
          <label htmlFor="cutaway-position">
            Move the cutaway plane (visual exploration only)
          </label>
          <input
            id="cutaway-position"
            type="range"
            min={-1.2}
            max={1.2}
            step={0.02}
            value={cutPosition}
            onChange={(event) => setCutPosition(Number(event.target.value))}
            style={{ display: "block", width: "100%" }}
          />
          <p className="form-intro">
            Moving this plane reveals part of the mesh. It does not
            represent lesion depth or tissue invasion.
          </p>
        </div>
      )}

      <div
        ref={mountRef}
        style={{
          width: "100%",
          height: 420,
          overflow: "hidden",
          borderRadius: 12,
          border: "1px solid var(--border)",
        }}
        aria-label="Interactive 3D anatomical model"
      />

      <p className="form-intro" role="status">
        {status}
      </p>
      <p className="form-intro">
        Research and education only. No lesion is marked or segmented. The cutaway is for anatomical exploration only and does not estimate lesion depth or tissue invasion.
        
      </p>
    </section>
  );
}
