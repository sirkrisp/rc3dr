import {
    AxesHelper, 
    GridHelper, 
    DirectionalLight,
    AmbientLight,
    PointLight,
    BoxGeometry,
    CylinderGeometry,
    SphereGeometry,
    CircleGeometry,
    Mesh,
    MeshPhongMaterial,
    MeshBasicMaterial,
    MeshDepthMaterial,
    Scene,
    PerspectiveCamera,
    WebGLRenderer,
    Matrix4,
    Vector3,
    Quaternion,
    LoadingManager,
    CameraHelper,
    LineBasicMaterial,
    EdgesGeometry,
    LineSegments,
} from './extern/three/build/three.module.js';

import * as THREE from './extern/three/build/three.module.js';

import {OBJLoader} from "./extern/three/examples/jsm/loaders/OBJLoader.js";

import { OrbitControls } from "./extern/three/examples/jsm/controls/OrbitControls.js";

import { TransformControls } from "./extern/three/examples/jsm/controls/TransformControls.js";

import { URDFLoader } from './extern/urdf-loader/URDFLoader.js';

import { URDFManipulator } from './extern/urdf-loader/urdf-manipulator-element.js'

/*
Reference coordinate frames for THREE.js and ROS.
Both coordinate systems are right handed so the URDF is instantiated without
frame transforms. The resulting model can be rotated to rectify the proper up,
right, and forward directions

THREE.js
   Y
   |
   |
   .-----X
 ／
Z

ROS URDf
       Z
       |   X
       | ／
 Y-----.

*/

export class RC3DR{
    data = () => {
        return {
            nsp : io('/rc3dr'),

            scene : null,
            renderer : null,
            canvas_hidden : null,

            timestamp_control : 0,
            timestamp_depth : 0,

            robot : null,
            target_link_name : "ee", // end-effector link
            ee_joint_name : "u5ee",

            urdf_loader : null,
            obj_loader : null,


            fusion_trans : [0,0,0.5],
            fusion_quat : [0,0,0,0],
            

            // --------------------------------
            // robot control
            renderer_control : null,

            // nsp_control : io('/robot-control'),
            camera_control : null,
            canvas_control : null,
            
            control_objects : null,
            control_cube : null,
            axis_helper : null,
            grid_helper : null,
            camera_helper : null,

            orbit : null,
            control : null,
            
            do_q_iter : false,
            q_steps : null,
            q_iter : 0,

            views : null,
            traversal_line : null,

            // getters
            do_send_e_pose : false,
            e_pose_callback : null,

            // --------------------------------
            // depth camera
            // Only use one renderer with a hidden canvas, 
            // and then draw from the context of the renderer.
            // https://discourse.threejs.org/t/multiple-renderer-vs-multiple-canvas/3085/2
            // nsp_depth : io('/depth-camera'),
            camera_depth : null,
            depth_camera_near : 1.0,
            depth_camera_far : 5.0,
            depth_camera_fov : 60,
            depth_camera_z : -1.0,
            canvas_depth : null,

            depth_material : null,
            do_tsdf : false,

            // getters
            do_send_depth_buffer : false,
            depth_buffer_callback : null,

            do_send_depth_camera_intrinsics : false,
            depth_camera_intrinsics_callback : null,
        }
    }

    props = ["filename", "targetObjFilename", 'width', 'height']

    methods = {
        // =============================================
        // COMMON

        get_scene : function() { return this.scene },

        get_robot : function() { return this.robot },

        load_robot : async function(){
            if(this.robot != null) this.scene.remove(this.robot)
            this.robot = await new Promise((resolve, reject) => { this.urdf_loader.load(
                this.filename,                    // The path to the URDF within the package OR absolute
                resolve                           // The robot is loaded!
            )})
            this.scene.add(this.robot)
            console.log(this.robot)

            // fusion volume box
            let geo = new EdgesGeometry( new BoxGeometry( 1.5, 1.5, 1.5 ) )
            let mat = new LineBasicMaterial( { color: "#212F3D", linewidth: 3 } )
            let fusion_volume = new LineSegments( geo, mat )
            fusion_volume.name = "fusion_volume"
            // this.robot.links[this.target_link_name].children[0].attach(fusion_volume)
            console.log(this.ee_joint_name)
            console.log(this.robot.joints[this.ee_joint_name])
            this.robot.joints[this.ee_joint_name].attach(fusion_volume)
            fusion_volume.position.copy(new Vector3(this.fusion_trans[0],this.fusion_trans[1],this.fusion_trans[2]))
        },

        load_target_obj : async function(){
            console.log(this.targetObjFilename)
            let gltf = await new Promise((resolve, reject) => 
                this.obj_loader.load(this.targetObjFilename, 
                            resolve,
                            (xhr) => { console.log((xhr.loaded / xhr.total * 100) + '% loaded' ) }, 
                            reject))
            console.log("target gltf:")
            console.log(gltf)
            // Scale
            gltf.scale.set(0.015, 0.015, 0.015)
            let target_obj = gltf
            console.log(target_obj)
            target_obj.name = "target_obj"
            this.robot.links[this.target_link_name].children[0].attach(target_obj)
            console.log(this.scene.getObjectByName("target_obj"))
            target_obj.position.copy(new Vector3(0,0,0.5))
            console.log(this.robot.links[this.target_link_name])
        },

        render_loop : async function(timestamp){
            // render control scene
            let dt_control = timestamp - this.timestamp_control
            if (dt_control > 1000 / 30) {
                this.timestamp_control = timestamp
                this.render_control()

                if(this.do_q_iter){
                    if(this.q_iter < this.q_steps.length){
                        console.log("q_step")
                        let q = this.q_steps[this.q_iter]
                        this.forward(q)
                        this.q_iter++
                    }else{
                        this.q_iter = 0
                        this.do_q_iter = false
                    }
                }else{
                    await this.send_control_pose()
                }

                if(this.do_send_e_pose) this.send_e_pose()
            }

            // render depth
            let dt_depth = timestamp - this.timestamp_depth
            if(dt_depth > 1000 / 10){
                this.render_depth()

                if(this.do_tsdf) await this.tsdf_fusion()

                if(this.do_send_depth_buffer) await this.send_depth_buffer()
                if(this.do_send_depth_camera_intrinsics) this.send_depth_camera_intrinsics()
            }

            requestAnimationFrame(this.render_loop)
        },

        // =============================================
        // Robot control

        render_control : function(){
            this.renderer.setClearColor('#F8F9F9', 1)
            this.renderer.antialias = true
            this.renderer.setSize(this.canvas_control.clientWidth, this.canvas_control.clientHeight)
            this.canvas_control.width = this.canvas_hidden.width
            this.canvas_control.height = this.canvas_hidden.height

            let fusion_volume = this.scene.getObjectByName("fusion_volume")
            if(fusion_volume != null) fusion_volume.visible = true
            if(this.views != null) this.views.visible = true
            if(this.traversal_line != null) this.traversal_line.visible = true
            for (let index = 0; index < this.control_objects.length; index++) {
                this.control_objects[index].visible = true
            }

            this.renderer.render(this.scene, this.camera_control)

            if(fusion_volume != null) fusion_volume.visible = false
            if(this.views != null) this.views.visible = false
            if(this.traversal_line != null) this.traversal_line.visible = false
            for (let index = 0; index < this.control_objects.length; index++) {
                this.control_objects[index].visible = false
            }

            // draw rendered image to canvas
            this.canvas_control.getContext('2d').drawImage(this.renderer.domElement, 0, 0)
        },

        start_exploration : function(){
            this.nsp.emit("explore", {'w_T_e_des' : this.control_cube.matrix.elements}, this.set_q_steps)
        },

        forward : function(angles){
            for (const name in angles) {
                this.robot.setAngle(name, angles[name]);
            }
        },

        scatter : function(data){
            let color = data["color"]
            let points = data["points"]
            for (let index = 0; index < points.length; index++) {
                const point = points[index]
                let geometry = new SphereGeometry( 0.1, 5, 5 );
                let material = new MeshPhongMaterial({ color: color, shininess: 40 })
                let sphere = new Mesh( geometry, material );
                this.scene.add(sphere)
                sphere.position.copy(new Vector3(point[0], point[1], point[2]))
            }
        },

        sample_views : function(){
            this.nsp.emit("sample_views", {}, this.draw_views)
        },

        select_views : function(){
            this.nsp.emit("select_views", {}, this.draw_views)
        },

        traverse_views : function(){
            this.nsp.emit("traverse_views", {}, this.set_q_steps)
        },

        // remove_views : function(){
        //     if(this.views != null){
        //         console.log("remove views")
        //         for (let index = 0; index < this.views.length; index++) {
        //             let element = this.views[index];
        //             console.log(element)
        //             this.scene.remove(element)
        //         }
        //     }
        // },

        draw_views : function(data){
            console.log("draw views")
            // this.remove_views()
            this.robot.joints[this.ee_joint_name].remove(this.views)

            let views = data["views"]
            console.log(views.length)

            let view_group = new THREE.Group();

            for (let index = 0; index < views.length; index++) {
                let view = views[index];
                let material = new THREE.LineBasicMaterial({
                    color: 0x0000ff
                });
                
                let geometry = new THREE.Geometry();
                geometry.vertices.push(
                    // x axis
                    new Vector3( 0, 0, 0 ),
                    new Vector3( 0.1, 0, 0 ),
                    // y axis
                    new Vector3( 0, 0, 0 ),
                    new Vector3( 0, 0.1, 0 ),
                    // z axis
                    new Vector3( 0, 0, 0 ),
                    new Vector3( 0, 0, 0.1 )
                );

                let line = new THREE.LineSegments( geometry, material )
                view_group.add(line);
                let quaternion = new THREE.Quaternion();
                quaternion.setFromAxisAngle( new THREE.Vector3( view[3], view[4], view[5] ), view[6] );
                line.quaternion.copy(quaternion)
                line.position.copy(new Vector3(view[0], view[1], view[2]))                
            }

            this.robot.joints[this.ee_joint_name].attach(view_group)
            view_group.position.copy(new Vector3(0,0,0))
            view_group.quaternion.copy(new Quaternion(0,0,0,0))

            this.views = view_group
        },

        send_control_pose : async function(){
            // send control pose, compute inverse kinematics, forward angles
            let w_T_e_des = this.control_cube.matrix.elements
            let q = await new Promise(resolve => this.nsp.emit('inverse', { 'w_T_e_des' :  w_T_e_des}, resolve));
            this.forward(q)
        },

        // setters (server side)
        set_q_steps : function(data){
            if("traversal" in data){
                
                console.log("traversal")
                console.log(this.views)
                // if(this.traversal_line != null) this.scene.remove(this.traversal_line)
                if(this.traversal_line != null) this.robot.joints[this.ee_joint_name].remove(this.traversal_line)
                
                let views = data["views"]
                this.traversal_line = new THREE.Group()

                let material = new THREE.MeshBasicMaterial( {color: "#D44739"} );
                for (let index = 0; index < views.length; index++) {
                    let view = views[index];

                    let geometry = new THREE.SphereGeometry( 0.04, 3, 3 );
                    
                    let sphere = new THREE.Mesh( geometry, material );

                    this.traversal_line.add(sphere)
                    sphere.position.copy(new Vector3(view[0], view[1], view[2]))
                    let quat = new Quaternion()
                    quat.setFromAxisAngle(new Vector3(view[3], view[4], view[5]), view[6])
                    sphere.quaternion.copy(quat)                
                }

                this.robot.joints[this.ee_joint_name].attach(this.traversal_line)
                this.traversal_line.position.copy(new Vector3(0,0,0))
                this.traversal_line.quaternion.copy(new Quaternion())
                

                // // draw lines between views indexed by traversal
                // let material = new THREE.LineBasicMaterial({
                //     color: 0x0000ff
                // });
                // let geometry = new THREE.Geometry();
                // let traversal = data["traversal"]
                // console.log(traversal)
                // console.log(this.views)
                // for (let index = 0; index < traversal.length-1; index++) {
                //     const a = traversal[index];
                //     const b = traversal[index+1];
                //     let va = this.views.children[a]
                //     let vb = this.views.children[b]
                //     geometry.vertices.push(va.position, vb.position)                    
                // }
                // let line = new THREE.LineSegments( geometry, material );
                // this.robot.joints[this.ee_joint_name].attach(line)
                // line.position.copy(new Vector3(0,0,0))
                // line.quaternion.copy(new Quaternion(0,0,0,0))
                // this.traversal_line = line
            }
            this.do_q_iter = true
            this.q_iter = 0
            this.q_steps = data["q_steps"]
            console.log(this.q_steps)
        },

        // getters (server side)
        get_e_pose : function(data, callback){
            this.do_send_e_pose = true
            this.e_pose_callback = callback
        },

        send_e_pose : function(){
            let w_T_e = this.scene.getObjectByName("fusion_volume").matrixWorld.elements

            if(this.e_pose_callback != null) this.e_pose_callback({ "w_T_e" : w_T_e })
            else this.nsp.emit("e_pose", { "w_T_e" : w_T_e })

            this.do_send_e_pose = false
            this.e_pose_callback = null
        },


        // client event handlers

        dragging_changed : function(event) {
            this.orbit.enabled = !event.value
        },

        keydown_handler : function (event) {
            switch (event.keyCode) {
                case 81: // Q
                    this.control.setSpace(this.control.space === "local" ? "world" : "local");
                    break;
                case 17: // Ctrl
                    this.control.setTranslationSnap(100);
                    this.control.setRotationSnap(THREE.Math.degToRad(15));
                    break;
                case 87: // W
                    this.control.setMode("translate");
                    break;
                case 69: // E
                    this.control.setMode("rotate");
                    break;
                case 82: // R
                    // this.control.setMode( "scale" );
                    this.nsp.emit('reload', {})
                    break;
                case 187:
                case 107: // +, =, num+
                    this.control.setSize(this.control.size + 0.1);
                    break;
                case 189:
                case 109: // -, _, num-
                    this.control.setSize(Math.max(this.control.size - 0.1, 0.1));
                    break;
                case 88: // X
                    this.control.showX = !this.control.showX;
                    break;
                case 89: // Y
                    this.control.showY = !this.control.showY;
                    break;
                case 90: // Z
                    this.control.showZ = !this.control.showZ;
                    break;
                case 32: // Spacebar
                    this.control.enabled = !this.control.enabled;
                    break;
            }
        },

        keyup_handler : function (event) {
            switch (event.keyCode) {
                case 17: // Ctrl
                    this.control.setTranslationSnap(null);
                    this.control.setRotationSnap(null);
                    break;
            }
        },

        // =============================================
        // Depth Camera

        render_depth : function(){
            this.renderer.antialias = false
            this.renderer.setClearColor('#FFFFFF', 1)
            this.renderer.setSize(this.canvas_depth.clientWidth, this.canvas_depth.clientHeight)
            this.renderer.setSize(this.canvas_control.clientWidth, this.canvas_control.clientHeight)
            this.canvas_depth.width = this.canvas_hidden.width
            this.canvas_depth.height = this.canvas_hidden.height

            let ctx = this.renderer.getContext()

            // ================================================
            // Set all materials to depth material
            this.scene.overrideMaterial = this.depth_material

            // render scene
            this.renderer.render(this.scene, this.camera_depth)

            if(this.do_tsdf){


                // get pixels
                let pixels1 = new Uint8Array(this.canvas_hidden.width * this.canvas_hidden.height * 4)
                ctx.readPixels(0, 0, this.canvas_hidden.width, this.canvas_hidden.height, ctx.RGBA, ctx.UNSIGNED_BYTE, pixels1)

                // Now render robot arm only
                let target_obj = this.scene.getObjectByName("target_obj")
                if (typeof target_obj !== 'undefined') target_obj.visible = false
                let background = this.scene.getObjectByName("background")
                background.visible = false

                this.renderer.render(this.scene, this.camera_depth)

                if (typeof target_obj !== 'undefined') target_obj.visible = true
                background.visible = true

                // get pixels
                let pixels2 = new Uint8Array(this.canvas_hidden.width * this.canvas_hidden.height * 4)
                ctx.readPixels(0, 0, this.canvas_hidden.width, this.canvas_hidden.height, ctx.RGBA, ctx.UNSIGNED_BYTE, pixels2)
                
                // ================================================


                // If R1 depth == R2 depth => set depth to zero
                // TODO Instead of applying this update here, it would be faster 
                // to do that in the shader directly. However, in real application
                // not possible anyways.
                // NOTE In real application one has to define a region of uncertainty.

                let pixels = new Uint8Array(this.canvas_hidden.width * this.canvas_hidden.height * 4)
                for (let x = 0; x < this.canvas_hidden.width; x++) {
                    for (let y = 0; y < this.canvas_hidden.height; y++) {
                        let idx1 = y * this.canvas_hidden.width + x
                        let idx2 = (this.canvas_hidden.height - y) * this.canvas_hidden.width + x
                        for (let i = 0; i < 3; i++) {
                            let tmp1 = idx1 * 4 + i
                            let tmp2 = idx2 * 4 + i
                            if(pixels1[tmp1] == pixels2[tmp1]){
                                pixels[tmp2] = 255 // ones means zero distance
                            }else{
                                pixels[tmp2] = pixels1[tmp1]
                            }
                        }
                        pixels[idx2*4+3] = 255
                    }
                }
                pixels = Uint8ClampedArray.from(pixels);
                let im_data = new ImageData(pixels, this.canvas_hidden.width, this.canvas_hidden.height)

                // draw rendered image to canvas
                this.canvas_depth.getContext('2d').putImageData(im_data, 0, 0)
            }else{

                this.canvas_depth.getContext("2d").drawImage(this.renderer.domElement, 0, 0)
            }

            this.scene.overrideMaterial = null
        },

        start_tsdf : function(){
            console.log("start tsdf")
            this.do_tsdf = true
        },

        end_tsdf : function(){
            console.log("end tsdf")
            this.do_tsdf = false
        },

        tsdf_fusion : async function() {
            // end-effector pose
            let w_T_f = this.scene.getObjectByName("fusion_volume").matrixWorld.elements

            // depth buffer
            let blob = await new Promise(resolve => this.canvas_depth.toBlob(resolve))
            let depth_buffer = await blob.arrayBuffer()

            await this.nsp.emit("tsdf_fusion", {depth_buffer: depth_buffer, w_T_f : w_T_f})
        },

        // getters
        // -----------------------------
        get_depth_buffer : function(data, callback){
            this.do_send_depth_buffer = true
            this.depth_buffer_callback = callback
        },

        send_depth_buffer : async function(){
            // depth buffer
            let blob = await new Promise(resolve => this.canvas_depth.toBlob(resolve))
            let depth_buffer = await blob.arrayBuffer()

            if(this.depth_buffer_callback != null) {
                this.depth_buffer_callback({depth_buffer : depth_buffer})
            }else{
                this.nsp.emit("depth_buffer", {depth_buffer : depth_buffer})
            }

            this.do_send_depth_buffer = false
            this.depth_buffer_callback = null
        },

        // -----------------------------
        get_depth_camera_intrinsics : function(data, callback){
            this.do_send_depth_camera_intrinsics = true
            this.depth_camera_intrinsics_callback = callback
        },

        send_depth_camera_intrinsics : function(){
            let focal = this.camera_depth.getFocalLength() // in pixels
            let height = this.camera_depth.getFilmHeight()
            let width = this.camera_depth.getFilmWidth()

            if(this.depth_camera_intrinsics_callback != null){
                this.depth_camera_intrinsics_callback({focal : focal, height : height, width : width})
            }else{
                this.nsp.emit("depth_camera_intrinsics", {focal : focal, height : height, width : width})
            }

            this.do_send_depth_camera_intrinsics = false
            this.depth_camera_intrinsics_callback = null
        },


        // =======================================================
        // Synchronization

        // TODO make also e_T_f a syned variable
        update_e_T_f : function(){
            let tsdf_fusion = this.scene.getObjectByName("fusion_volume")
            let target_obj = this.scene.getObjectByName("target_obj")
            // rotate volume and object by pi around y axis
            let quat = new Quaternion()
            quat.setFromAxisAngle(new Vector3(0,1,0), Math.PI)
            quat = quat.multiply(tsdf_fusion.quaternion)

            tsdf_fusion.quaternion.copy(quat)
            target_obj.quaternion.copy(quat)
            
            this.nsp.emit("update_e_T_f", {quat : quat.toArray()})

        },

        sync_w_T_e_des : async function(){
            console.log("sync w_T_e")
            let data = await new Promise(resolve => this.nsp.emit("sync_var", {var_name : "w_T_e"}, resolve))
            data = data["value"]
            let trans = data["w_T_e_trans"]
            let quat = data["w_T_e_quat"]
            console.log(trans)
            console.log(quat)
            let trans2 = new Vector3(trans[0], trans[1], trans[2]);
            let quat2 = new Quaternion()
            let vec3 = new Vector3(quat[0], quat[1], quat[2])
            console.log(vec3)
            // quat2.setFromAxisAngle( vec3, quat[4]);
            console.log(quat[0], quat[1])
            quat2.setFromAxisAngle( new Vector3(quat[0], quat[1], quat[2]), quat[3] );
            console.log("Quaternion")
            console.log(quat2)
            console.log(this.control_cube.matrixWorld.elements)
            this.control_cube.quaternion.copy(quat2)
            this.control_cube.position.copy(trans2)
            console.log(this.control_cube.matrixWorld.elements)
        }
    }



    mounted = function() {
        // ==========================================
        // COMMON

        // create scene
        this.scene = new Scene();

        // light
        this.scene.add(new DirectionalLight(0xffffff, 0.5)) // color, intensity, position default = top
        this.scene.add(new AmbientLight(0x404040))
        this.scene.add(new PointLight(0xffffff, 0.8, 100, 2))

        // add background box
        let geometry = new THREE.BoxGeometry( 40, 40, 0.5 );
        let material = new MeshPhongMaterial( {color : "#3498DB", shininess : 40, opacity: 0.2, transparent: true})
        let cube = new THREE.Mesh( geometry, material );
        cube.name = "background"
        this.scene.add( cube );
        cube.position.copy(new Vector3(0,0,this.depth_camera_far-0.5 + this.depth_camera_z))

        // Renderer
        this.canvas_hidden = this.$refs.canvasHidden
        this.renderer = new WebGLRenderer({ canvas: this.canvas_hidden })
        this.renderer.setPixelRatio( window.devicePixelRatio ) // NOTE: setPixelRatio changes canvas width and height
        // some more render properties
        this.renderer.gammaOutput = true
        this.renderer.gammaFactor = 2.2
        this.renderer.setClearColor('#F8F9F9', 1)
        
        // Load robot
        // https://www.npmjs.com/package/urdf-loader
        let manager = new LoadingManager();
        this.urdf_loader = new URDFLoader(manager);
        this.load_robot()        

        // Load target
        this.obj_loader = new OBJLoader();
        this.load_target_obj()

        // ==========================================================
        // Robot control

        // canvas
        this.canvas_control = this.$refs.canvasControl
        console.log(this.canvas_control)
        
        // control camera
        this.camera_control = new PerspectiveCamera(75, this.canvas_control.clientWidth / this.canvas_control.clientHeight, 0.1, 1000)
        this.camera_control.position.y = 5

        // axis
        this.axis_helper = new AxesHelper(5)
        this.scene.add(this.axis_helper)

        // grid
        this.grid_helper = new GridHelper(10, 10)
        this.scene.add(this.grid_helper)

        // depth camera helper
        let depth_camera = new PerspectiveCamera(this.depth_camera_fov, this.canvas_control.clientWidth / this.canvas_control.clientHeight, this.depth_camera_near, this.depth_camera_far)
        // depth camera pose
        let trans = new Vector3(0, 3, this.depth_camera_z);
        let quat = (new Quaternion()).setFromAxisAngle(new Vector3(0, 1, 0), Math.PI);
        this.scene.add(depth_camera)
        depth_camera.position.copy(trans)
        depth_camera.quaternion.copy(quat)
        this.camera_helper = new CameraHelper(depth_camera)
        this.scene.add(this.camera_helper)
        

        // control cube
        this.control_cube = new Mesh(new BoxGeometry(1, 1, 1), 
                                    new MeshPhongMaterial(
                                        { color: '#7FB3D5', shininess: 40, opacity: 0.5, transparent: true, depthTest: false }
                                    ));
        this.scene.add(this.control_cube)
        this.control_cube.position.copy(new Vector3(0, 3, 1.5))
        quat = new Quaternion(0,1,0,Math.PI)
        quat.setFromAxisAngle(new THREE.Vector3( 0, 1, 0 ), Math.PI)
        this.control_cube.quaternion.copy(quat)

        // scene orbit control
        this.orbit = new OrbitControls(this.camera_control, this.canvas_control)
        this.orbit.update()
        // TODO might cause problems
        this.orbit.addEventListener( 'change', this.render_control )

        // end effector control
        this.control = new TransformControls(this.camera_control, this.canvas_control)
        this.control.setSize(2)
        this.control.attach(this.control_cube)
        this.control.addEventListener('dragging-changed', this.dragging_changed)
        // this.control.addEventListener('objectChange', this.send_control_pose)
        this.scene.add(this.control)
        
        // gather control objects
        this.control_objects = [this.axis_helper, this.grid_helper, this.control_cube, this.control, this.camera_helper]

        // handlers
        window.addEventListener('keydown', this.keydown_handler)
        window.addEventListener('keyup', this.keyup_handler)

        // io handlers
        this.nsp.on("get_e_pose", this.get_e_pose)
        this.nsp.on("set_q_steps", this.set_q_steps)
        this.nsp.on("scatter", this.scatter)
        
        // ========================================================
        // Depth camera
        
        this.canvas_depth = this.$refs.canvasDepth
        
        // depth camera
        this.camera_depth = new PerspectiveCamera(this.depth_camera_fov, this.canvas_depth.clientWidth / this.canvas_depth.clientHeight, this.depth_camera_near, this.depth_camera_far)
        // depth camera pose
        this.camera_depth.position.copy(trans)
        this.camera_depth.quaternion.copy(quat)

        // custom depth material
        // https://github.com/mrdoob/three.js/blob/master/examples/webgl_depth_texture.html
        this.depth_material = new THREE.ShaderMaterial( {
            vertexShader: this.$refs.vertexshader.textContent.trim(),
            fragmentShader: this.$refs.fragmentshader.textContent.trim(),
            uniforms: {
                cameraNear: { value: this.camera_depth.near },
                cameraFar: { value: this.camera_depth.far },
            }
        } );

        // io handlers
        this.nsp.on("get_depth_buffer", this.get_depth_buffer)
        this.nsp.on("get_depth_camera_intrinsics", this.get_depth_camera_intrinsics)
        this.nsp.on("start_tsdf", this.start_tsdf)
        this.nsp.on("end_tsdf", this.end_tsdf)

        // start render loop
        this.render_loop()
    }

    template = `
        <div class="dash-card-container">
            <canvas :style="{display: 'none'}" ref="canvasHidden"></canvas>

            <div class="dash-card">
                <div class="dash-card-title">Robot control</div>
                <canvas :style="{width : width + 'px', height : height + 'px'}" ref="canvasControl"></canvas>
            </div>

            <div class="dash-card">
                <div class="dash-card-title">Depth Camera</div>

                <canvas :style="{width : width + 'px', height : height + 'px'}" ref="canvasDepth"></canvas>
                <button v-on:click="do_tsdf = !do_tsdf">
                    <div v-if="do_tsdf">
                        Stop TSDF
                    </div>
                    <div v-else>
                        Start TSDF
                    </div>
                </button>

                <script type="x-shader/x-vertex" ref="vertexshader">
                    varying vec2 vUv;
                    varying float depth;
                    varying float ndc_depth;
                    
                    void main() {
                        vUv = uv;
                        vec4 tmp = modelViewMatrix * vec4(position, 1.0);
                        gl_Position = projectionMatrix * tmp;
                        // NOTE Measure depth in positive z-direction
                        depth = -tmp.z;
                        ndc_depth = gl_Position.z / gl_Position.w;
                    }
                </script>
                
                <script type="x-shader/x-fragment" ref="fragmentshader">
                    #include <packing>

                    varying float depth;
                    varying float ndc_depth;
                    uniform float cameraNear;
                    uniform float cameraFar;

                    void main() {
                        // TODO Just use projection matrix

                        // eye coords
                        float z_e = depth; 

                        // normalized device coords
                        float A = (cameraFar + cameraNear) / (cameraFar - cameraNear);
                        float B = -2.0*cameraFar*cameraNear / (cameraFar - cameraNear);
                        float z_n = (A*z_e + B*1.0) / z_e;

                        
                        // float d = z_eye / cameraFar;

                        float d = (1.0 - z_n) / 2.0;

                        gl_FragColor.rgb = vec3( d );
                        gl_FragColor.a = 1.0;
                    }
                </script>
            </div>

            <div class="dash-card">
                <div class="dash-card-title">Actions</div>
                <button v-on:click="start_exploration">Start exploration</button>
                <button v-on:click="sample_views">Sample views</button>
                <button v-on:click="select_views">Select views</button>
                <button v-on:click="traverse_views">Traverse views</button>
                <button v-on:click="update_e_T_f">Flip volume</button>
                <button v-on:click="do_q_iter = true">Re-iterate</button>
                <button v-on:click="sync_w_T_e_des" style="display:none">Sync w_T_e_des</button>
                
            </div>
        
        </div>
    `
}












