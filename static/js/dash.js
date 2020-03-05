
import Vue from "./extern/vue.esm.browser.js"
import {RC3DR } from "./rc3dr.js"

var socket = io.connect();

export class Dash{
    constructor(){
        // setup connection
        socket.on('connect', function () {
            console.log("Dash is connected.")
            console.log(this)
            socket.emit('client_connected', {data: 'I\'m connected!'})
        });
        socket.on('c1_test', function (msg) {
            console.log(msg);
        });


        // vue app
        new Vue({ 
            el: '#root',
            data : {
                socket : socket
            },

            template:`
            <div class="vue-root">
                <div class="dash-title">RC3DR - Robot-controlled 3D reconstruction:</div>
                <div class="dash">
                    <robot-card filename="/media/models/urdf/6dof_macro.urdf" target-obj-filename="/media/models/obj/cow.obj" width="500" height="500">
                    </robot-card>
                </div>

            </div>
            `,
            components : {
                'robot-card' : new RC3DR(),
            }
        })
    }
}

const dash = new Dash();
console.log(dash);


