import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';
const names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];
const latest={}, renderers=[];
function makeViewer(id,pose,tint){
 const canvas=document.getElementById(id),renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x0d1015);
 const scene=new THREE.Scene();scene.add(new THREE.HemisphereLight(0xffffff,0x455160,2.2));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(1,-1,2);scene.add(light);const grid=new THREE.GridHelper(.65,20,0x536879,0x263744);grid.rotation.x=Math.PI/2;scene.add(grid);scene.add(new THREE.AxesHelper(.09));
 const camera=new THREE.PerspectiveCamera(40,1,.001,4);camera.up.set(0,0,1);camera.position.set(.42,-.48,.35);const controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.target.set(0,0,.10);let model;
 new URDFLoader().load('/so101/so101_new_calib.urdf',m=>{model=m;model.ignoreLimits=true;model.traverse(o=>{if(o.isMesh){o.material=o.material.clone();o.material.color.setHex(tint)}});scene.add(model)});const resize=()=>{const r=canvas.getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix()};new ResizeObserver(resize).observe(canvas);resize();
 renderers.push(()=>{if(model){const p=pose();for(const n of names)if(model.joints[n]&&Number.isFinite(p[n]))model.setJointValue(n,p[n]);model.updateMatrixWorld(true)}controls.update();renderer.render(scene,camera)});
}
makeViewer('measuredView',()=>latest,0x7de2c3);makeViewer('commandedView',()=>{const p={...latest},g=latest.servo_goal||[];names.forEach((n,i)=>{if(Number.isFinite(g[i]))p[n]=g[i]});return p},0xa9d8ff);
window.addEventListener('robot-message',({detail:m})=>{const s=m.state||m.data;if(s)Object.assign(latest,s)});function frame(){renderers.forEach(r=>r());requestAnimationFrame(frame)}frame();
