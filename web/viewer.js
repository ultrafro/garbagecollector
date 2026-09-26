import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';

const canvas = document.getElementById('ikView');
const status = document.getElementById('ikStatus');
const xyzStatus=document.createElement('pre');xyzStatus.id='ikXYZ';xyzStatus.style.cssText='white-space:pre-wrap;font-size:12px';status.after(xyzStatus);
status.style.cssText = 'height:auto;white-space:normal;text-align:left;line-height:1.5';
canvas.style.touchAction = 'none';
const help = document.createElement('div');
const phaseStatus=document.createElement('div');phaseStatus.style.cssText='padding:8px;color:#ffbd70;white-space:normal';canvas.after(phaseStatus);
help.innerHTML = '<button id="reset3d">Reset view</button> Drag: orbit · scroll: zoom · right-drag: pan · gold: measured · mint: commanded';
canvas.before(help);
let robot, commanded, startRobot, latest={}, debug;
let renderer;
try { renderer = new THREE.WebGLRenderer({canvas, antialias:true}); }
catch(error) { status.textContent = `3D renderer unavailable: ${error.message}`; throw error; }
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0x111923);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40,1,.001,20);
camera.up.set(0,0,1);
const controls = new OrbitControls(camera,canvas);
controls.enableDamping=true;
controls.minDistance=.12; controls.maxDistance=3;
const reset=()=>{camera.position.set(.42,-.46,.38);controls.target.set(0,0,.13);controls.update();};
reset(); document.getElementById('reset3d').onclick=reset;
scene.add(new THREE.HemisphereLight(0xffffff,0x596575,2.5));
const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(1,-1,2);scene.add(light);
const grid=new THREE.GridHelper(1.2,24,0x597183,0x293b49);grid.rotation.x=Math.PI/2;scene.add(grid);
// Ground reference used by the dive: the calibrated URDF floor is -57 mm;
// the user-selected offset moves this translucent plane up/down from there.
let groundOffsetCm=1.8;
const groundPlane=new THREE.Mesh(new THREE.PlaneGeometry(.9,.9),new THREE.MeshBasicMaterial({color:0x35e0c0,transparent:true,opacity:.58,side:THREE.DoubleSide,depthWrite:false}));
groundPlane.position.z=-.057+groundOffsetCm/100;groundPlane.renderOrder=2;scene.add(groundPlane);
const groundEdge=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(.9,.9)),new THREE.LineBasicMaterial({color:0x7dffe5,transparent:true,opacity:1,depthTest:false}));
groundEdge.position.copy(groundPlane.position);groundEdge.renderOrder=3;scene.add(groundEdge);
const referencePlane=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(.9,.9)),new THREE.LineDashedMaterial({color:0xff668e,dashSize:.025,gapSize:.018,depthTest:false}));
referencePlane.position.z=-.057;referencePlane.renderOrder=3;scene.add(referencePlane);
const groundReference=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-.48,0,-.057),new THREE.Vector3(-.48,0,groundPlane.position.z)]),new THREE.LineDashedMaterial({color:0xff668e,dashSize:.018,gapSize:.012,depthTest:false}));
groundReference.computeLineDistances();groundReference.renderOrder=4;scene.add(groundReference);
const groundLabelCanvas=document.createElement('canvas');groundLabelCanvas.width=640;groundLabelCanvas.height=96;
const groundLabelContext=groundLabelCanvas.getContext('2d');groundLabelContext.fillStyle='#063d39';groundLabelContext.fillRect(0,0,640,96);groundLabelContext.font='bold 34px sans-serif';groundLabelContext.fillStyle='#7dffe5';groundLabelContext.fillText('GRAB HEIGHT PLANE',18,43);groundLabelContext.font='28px sans-serif';groundLabelContext.fillText('+1.8 cm',18,78);
const groundLabel=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(groundLabelCanvas),depthTest:false,sizeAttenuation:false}));groundLabel.scale.set(.28,.055,1);groundLabel.position.set(-.72,0,groundPlane.position.z+.045);groundLabel.renderOrder=5;scene.add(groundLabel);
scene.add(new THREE.AxesHelper(.12));
const startAxes=new THREE.AxesHelper(.07), liveAxes=new THREE.AxesHelper(.05);
startAxes.visible=false; liveAxes.visible=false;scene.add(startAxes,liveAxes);
const arrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(),.15,0xff668e,.025,.012);
arrow.visible=false;scene.add(arrow);
// The long ray shows direction only; it is not a commanded reach distance.
const ray=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineDashedMaterial({color:0xff668e,dashSize:.014,gapSize:.018,depthTest:false}));
ray.visible=false;ray.renderOrder=10;scene.add(ray);
const origin=new THREE.Mesh(new THREE.SphereGeometry(.012,24,16),new THREE.MeshBasicMaterial({color:0x62cfff,depthTest:false}));
origin.visible=false;origin.renderOrder=11;scene.add(origin);
const labelCanvas=document.createElement('canvas');labelCanvas.width=512;labelCanvas.height=80;
const labelContext=labelCanvas.getContext('2d');labelContext.fillStyle='#092a3e';labelContext.fillRect(0,0,512,80);
labelContext.font='bold 30px sans-serif';labelContext.fillStyle='#8bdeff';labelContext.fillText('INITIAL POSE / START',18,50);
const label=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(labelCanvas),depthTest:false,sizeAttenuation:false}));
label.scale.set(.32,.05,1);label.visible=false;label.renderOrder=12;scene.add(label);
const legend=document.createElement('div');legend.textContent='Blue: captured initial pose · pink dotted ray: intended direction (1 m extension, not commanded travel) · gold trail: measured motion';help.after(legend);
const fitButton=document.createElement('button');fitButton.textContent='Fit trajectory';help.append(fitButton);
fitButton.onclick=()=>{if(!ray.visible)return;const center=startAxes.position.clone().addScaledVector(arrow.getWorldDirection(new THREE.Vector3()),0);const end=new THREE.Vector3().fromBufferAttribute(ray.geometry.attributes.position,1);center.add(end).multiplyScalar(.5);controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(1,-1,.8));controls.update();};
const trace=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xffbd70}));scene.add(trace);
let trail=[], signature='';
const setPose=(model,pose)=>{if(!model||!pose)return;for(const [name,value] of Object.entries(pose))if(model.joints[name]&&Number.isFinite(value))model.setJointValue(name,value);model.updateMatrixWorld(true);};
const tip=model=>model.links.gripper_frame_link||model.links.gripper_link;
const ghost=(model,color,opacity)=>{const clone=model.clone(true);clone.traverse(o=>{if(o.isMesh)o.material=new THREE.MeshBasicMaterial({color,wireframe:true,transparent:true,opacity,depthWrite:false});});clone.visible=false;scene.add(clone);return clone;};
const manager=new THREE.LoadingManager();
let failed=[];
manager.onError=url=>{failed.push(url);status.textContent=`Model asset failed: ${url}`;};
manager.onLoad=()=>{if(!robot)return;commanded=ghost(robot,0x7de2c3,.3);startRobot=ghost(robot,0x62cfff,.4);let count=0;robot.traverse(o=>{if(o.isMesh)count++;});canvas.dataset.meshCount=count;canvas.dataset.ready='true';update();};
const loader=new URDFLoader(manager);
status.textContent='Loading SO-101 URDF and STL meshes…';
loader.load('/so101/so101_new_calib.urdf',model=>{robot=model;robot.ignoreLimits=true;scene.add(robot);setPose(robot,latest);},undefined,error=>{status.textContent=`SO-101 load failed: ${error.message}`;});
function rayLine(d){
 if(d.progress==null)return '';
 return `
Approach: ${(d.progress*1000).toFixed(0)} mm along the forward vector${d.reaim_deg?` · steered ${d.reaim_deg.toFixed(1)}°`:``}`;
}
function contactLine(d){
 if(!d.contact_loads)return '\nContact: no servo load telemetry';
 const w=d.contact_watched||'wrist_flex',sm=d.contact_smoothed,thr=d.contact_threshold,ma=d.contact_smoothing||3;
 const head=d.contact_joint?`CONTACT on ${w}`
  :!d.contact_armed?`disarmed until ${(d.contact_arm_mm||0).toFixed(0)} mm travel (at ${(d.contact_travel_mm||0).toFixed(0)} mm)`
  :`${w} ${sm==null?'—':sm.toFixed(0)}/${thr==null?'—':thr.toFixed(0)} (MA${ma}) · ${d.contact_samples||0}/3 samples`;
 const rest=Object.keys(d.contact_loads).filter(n=>n!==w)
  .map(n=>`${n} ${d.contact_loads[n]==null?'—':d.contact_loads[n].toFixed(0)}`).join(' · ');
 return `\nContact: ${head}\n  other joints (not armed to stop): ${rest}`;
}
function update(){
 if(debug?.commanded_position){const xyz=v=>v.map(n=>(n*1000).toFixed(1)).join(', ');xyzStatus.textContent=`XYZ (mm, URDF base frame; measured = encoder-derived FK)\nStart:     ${xyz(debug.start_position)}\nCommanded: ${xyz(debug.commanded_position)}\nMeasured:  ${xyz(debug.measured_position)}\nXYZ gap: ${((debug.position_error_m||0)*1000).toFixed(1)} mm · travel along initial ray: commanded ${((debug.commanded_along_ray_m||0)*1000).toFixed(1)} / measured ${((debug.along_ray_m||0)*1000).toFixed(1)} mm` + rayLine(debug) + contactLine(debug);}
 if(!robot)return;
 setPose(robot,latest);
 const liveTip=tip(robot);liveTip.getWorldPosition(liveAxes.position);liveTip.getWorldQuaternion(liveAxes.quaternion);liveAxes.visible=true;
 if(debug?.start_joints&&commanded&&startRobot){
  setPose(commanded,debug.commanded_joints);setPose(startRobot,debug.start_joints);commanded.visible=true;startRobot.visible=true;
  const startTip=tip(startRobot);startTip.getWorldPosition(startAxes.position);startTip.getWorldQuaternion(startAxes.quaternion);startAxes.visible=true;
  const key=JSON.stringify(debug.start_joints);if(key!==signature){signature=key;trail=[];}
  const direction=new THREE.Vector3(...debug.forward);const length=direction.length();
  arrow.position.copy(startAxes.position);if(length>1e-6){arrow.setDirection(direction.normalize());arrow.setLength(.15,.025,.012);arrow.visible=true;}
  origin.position.copy(startAxes.position);origin.visible=true;label.position.copy(startAxes.position).add(new THREE.Vector3(0,0,.08));label.visible=true;
  if(length>1e-6){ray.geometry.dispose();ray.geometry=new THREE.BufferGeometry().setFromPoints([startAxes.position.clone(),startAxes.position.clone().addScaledVector(direction,1)]);ray.computeLineDistances();ray.visible=true;canvas.dataset.trajectory='visible';}
  const p=liveAxes.position.clone();if(!trail.length||p.distanceTo(trail.at(-1))>.0005){trail.push(p);if(trail.length>1000)trail.shift();trace.geometry.dispose();trace.geometry=new THREE.BufferGeometry().setFromPoints(trail);}
 }
 status.textContent=failed.length?`Some meshes failed to load: ${failed.join(', ')}`:`SO-101 · ${canvas.dataset.meshCount||'loading'} meshes · ground offset ${groundOffsetCm.toFixed(1)} cm${debug ? ` · wrist +Z ray · commanded ${(debug.progress*1000).toFixed(1)} mm · measured ${((debug.along_ray_m||0)*1000).toFixed(1)} mm · cross-track ${((debug.cross_track_m||0)*1000).toFixed(1)} mm · target ${(debug.target_width*100).toFixed(1)}% · joint error ${debug.deviation} rad` : ' · awaiting grab capture'} — URDF zero calibration is not yet verified against hardware.`;
}
window.addEventListener('robot-message',({detail:m})=>{const state=m.state||m.data;if(state)latest={...latest,...state};if(Number.isFinite(m.ground_offset_cm)){groundOffsetCm=m.ground_offset_cm;groundPlane.position.z=-.057+groundOffsetCm/100;groundEdge.position.copy(groundPlane.position);groundReference.geometry.dispose();groundReference.geometry=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-.48,0,-.057),new THREE.Vector3(-.48,0,groundPlane.position.z)]);groundReference.computeLineDistances();groundLabel.position.z=groundPlane.position.z+.045;groundLabelContext.clearRect(0,0,640,96);groundLabelContext.fillStyle='#063d39';groundLabelContext.fillRect(0,0,640,96);groundLabelContext.fillStyle='#7dffe5';groundLabelContext.font='bold 34px sans-serif';groundLabelContext.fillText('GRAB HEIGHT PLANE',18,43);groundLabelContext.font='28px sans-serif';groundLabelContext.fillText(`${groundOffsetCm>=0?'+':''}${groundOffsetCm.toFixed(1)} cm`,18,78);groundLabel.material.map.needsUpdate=true;}if(m.grab_debug)debug=m.grab_debug;if(m.grab_message)phaseStatus.textContent=m.grab_message;if(m.type==='notice'||m.type==='error')phaseStatus.textContent=m.message;update();});
const observer=new ResizeObserver(()=>{const bounds=canvas.getBoundingClientRect();renderer.setSize(bounds.width,bounds.height,false);camera.aspect=bounds.width/bounds.height;camera.updateProjectionMatrix();});observer.observe(canvas);
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
