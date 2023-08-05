import Vue from 'vue'
import Vuex from 'vuex'
import axios from 'axios'
import * as d3 from "d3";
import * as Global from "../plugins/global";
import { interpolate } from 'd3';


axios.defaults.headers.common['Access-Control-Allow-Origin'] = '*';

//mount Vuex
Vue.use(Vuex)

function rnd( seed ){
    seed = ( seed * 9301 + 49297 ) % 233280; 
    return seed / ( 233280.0 );
}

function rnd_generate(start, end, num){
    let originalArray = [];
    for (let i = start; i <= end; i++){
        originalArray.push(i);
    }
    let sampled_array = Global.deepCopy(originalArray);
    originalArray.sort(function(){ return 0.5 - Math.random(); });
    return [sampled_array, originalArray.slice(0, num)];
}

//create VueX
const store = new Vuex.Store({
    state:{ // data
        server_url: 'http://localhost:30221',
        manifest_data: [],
        class_info: [{"num": 8, "id": 0, "full_name": "test1"}, 
            {"num": 4, "id": 1, "full_name": "test2"},
            {"num": 6, "id": 2, "full_name": "test3"},
            {"num": 5, "id": 3, "full_name": "test4"},
            {"num": 8, "id": 4, "full_name": "test5"},
            {"num": 6, "id": 5, "full_name": "test6"},
            {"num": 6, "id": 6, "full_name": "test7"},
            {"num": 5, "id": 7, "full_name": "test8"},
            {"num": 7, "id": 8, "full_name": "test9"},
            {"num": 9, "id": 9, "full_name": "test10"},
            {"num": 4, "id": 10, "full_name": "test11"},
            {"num": 6, "id": 11, "full_name": "test12"},
            {"num": 4, "id": 12, "full_name": "test13"},
            {"num": 7, "id": 13, "full_name": "test14"},
            {"num": 9, "id": 14, "full_name": "test15"},
            {"num": 7, "id": 15, "full_name": "test16"},
            {"num": 5, "id": 16, "full_name": "test17"},
            {"num": 4, "id": 17, "full_name": "test18"},
            {"num": 6, "id": 18, "full_name": "test19"},
        ],
        hierarchy_data: null,
        data_id_mapping: null,
        selected_class: 10,
        selected_frames: [],
        selected_ids: [],
        
        selected_cluster: {cluster_id: -1, selected_action: null},
        // selected_cluster: {cluster_id: "M3-11", selected_action: "158-598"},
        anchor_alignments: {},
        expand_node_ids: [],
        selected_details: [],
        selected_action: {},
        selected_video: {},
        key_frames: [],
        
        // for update
        history: [],
        constraint: {},
        update_type: "comparison",

        // work history
        work_history: {},
        in_length_compare: false,
        compare_step: -1
    },
    mutations:{ // function to modify the state
        set_manifest_data(state, manifest_data){
            console.log("set manifest data");
            state.manifest_data = manifest_data;
            state.class_names = manifest_data.class_name;
            let m = {
                'BaseballPitch': "baseball",
                'BasketballDunk': "basketball",
                'Billiards': "billiards",
                'CleanAndJerk': "weightlifting",
                'CricketBowling': "Bowling",
                'CricketShot': "shoting",
                'Diving': "diving",
                'FrisbeeCatch': "frisbee",
                'GolfSwing': "golf",
                'HammerThrow': "hammer",
                'HighJump': "high jump",
                'JavelinThrow': "Javelin",
                'LongJump': "long jump",
                'PoleVault': "pole vault",
                'Shotput': "shotput",
                'SoccerPenalty': "soccer",
                'TennisSwing': "tennis",
                'ThrowDiscus': "discus",
                'VolleyballSpiking': "volleyball"
            };
            // for (let i = 0; i < state.class_names.length; i++){
            //     let tmp = state.class_names[i];
            //     if (tmp in m){
            //         state.class_names[i] = m[tmp];
            //     }
            // }
        },
        set_hierarchy(state, hierarchy_data){
            console.log("set hierarchy data");

            state.data_id_mapping = {};
            hierarchy_data = d3.hierarchy(hierarchy_data,
                function(d){
                    let children = d.children;
                    return children;
                });
            hierarchy_data.eachAfter(e => {
                e.id = e.data.name;
                state.data_id_mapping[e.id] = e;
                e.aligns = e.data.aligns;
                e.y_layout_test = Global.deepCopy(e.data.y_layout);
                e.y_layout = e.data.y_layout;

                // all_children: all children
                // children: children that are visible
                // _children: children that are invisible
                e.all_children = e.children;
                if(e.children) e.children.forEach((d,i) => d.siblings_id = i);
                if(e.children) e.children.forEach(d => d.parent = e)
                e._children = [];
                e._total_width = 0;
                e.api = 0;
                e.mismatch_scores = e.data.mismatch_scores;
                e.rep_cols = e.data.rep_cols;
                e.rep_rows = e.data.rep_rows;
            });
            hierarchy_data.eachAfter(e => {
                if (!e.all_children){ // leaf nodes
                    let parent = e.parent;
                    let siblings_id = parent.all_children.indexOf(e);
                    e.mat = parent.aligns[siblings_id].filter(d => d >= 0).map(d => [d]);
                    e.actions = [e.data.name];
                    e.rep_action = e.data.name; // a leaf node itself is its representative action
        
                    e.vid = e.data.video_id;
                    e.start_idx = e.data.bound[0];
                    e.end_idx = e.data.bound[1];

                    let all_frames = [];
                    for (let i = e.start_idx; i <= e.end_idx; i++){
                        all_frames.push(i);
                    }
                    e.all_frames = all_frames;
                    // TODO: it does not make sense to slice 3 out of more than 3 representative images.
                    // restrictly generating 3 representative images.
                    e.rep_frames = e.data.key_frames; 
                    e.rep_frames = e.rep_frames.map(d => e.vid + "-" + d);
                    e.all_frames = e.all_frames.map(d => e.vid + "-" + d);

                    e.single_frame = e.data.single_frame - e.data.bound[0];
                    e.labels = new Array(e.mat.length).fill([-1]);
                    e.labels[e.data.single_frame - e.data.bound[0]] = [1];
                    e.y_layout_height = new Array(e.mat.length).fill(1.0 / 10);
                    e.action_lengths = [e.all_frames.length]
                }
                else{
                    let mats = [];
                    let labels = [];
                    let aligns = Global.deepCopy(e.aligns);
                    aligns.forEach(d => {
                        let count = 0;
                        for (let i = 0; i < d.length; i++){
                            if (d[i] > -1) d[i] = count++;
                        }
                    });
                    // console.log("aligns", aligns);
                    for (let i = 0; i < aligns[0].length; i++){
                        let mat = [];
                        let label = [];
                        for (let j = 0; j < aligns.length; j++){
                            if (aligns[j][i] === -1){
                                let tmp = new Array(e.all_children[j].mat[0].length).fill(-1);
                                mat = mat.concat(tmp);
                                label = label.concat(tmp);
                            }
                            else{
                                mat = mat.concat(e.all_children[j].mat[aligns[j][i]]);
                                label = label.concat(e.all_children[j].labels[aligns[j][i]]);
                            }
                        }
                        mats.push(mat);
                        labels.push(label);
                    }
                    // console.log('labels:', labels);

                    e.mat = mats;
                    e.labels = labels;
                    // e.actions = e.all_children[0].actions.concat(e.all_children[1].actions);
                    e.actions = e.all_children.map(d => d.actions).reduce((p,v) => p.concat(v), []);

                    // e.aligns.forEach((d, i) => {
                    //     let flag = 0;
                    //     d.forEach((f, j) => {
                    //         if (f !== -1 && flag === 0) flag = 1; 
                    //         if (flag === 0) e.y_layout[i][j] = 0; 
                    //     })
                    //     flag = 0;
                    //     for (let j = d.length - 1; j >= 0; j--){
                    //         if (d[j] !== -1 && flag === 0) flag = 1;
                    //         if (flag === 0) e.y_layout[i][j] = 0;
                    //     }
                    // })

                    // // TODO: careful
                    // if (e.all_children[0].all_children) {
                    //     e.aligns.forEach((d, i) => {
                    //         d.forEach((f, j) => {
                    //             e.y_layout[i][j] *= 2; 
                    //         })
                    //     })
                    // }
                    // e.y_layout_max = e.y_layout[0].map(
                    //     (d, i) => e.y_layout.reduce((p,v) => Math.max(p, v[i]), 0));
                    e.y_layout_height = e.y_layout[e.y_layout.length - 1];
                    // e.y_layout_height = e.y_layout_height.map(d => d / (e.y_layout.length));
                    
                    e.rep_action_id = e.data.center;
                    e.rep_action = state.data_id_mapping[e.rep_action_id];
                    e.rep_frames = e.rep_action.rep_frames;
                    e.action_lengths = e.all_children.map(d => d.action_lengths).reduce((p,v) => p.concat(v), []);
                }
                e.rep_action_position = e.data.center_align;
                e.actions_map = {}
                e.actions.forEach(n => e.actions_map[n] = 1);
            })
            hierarchy_data.all_descendants = hierarchy_data.descendants();
            hierarchy_data.eachBefore((d, i) => d.order = i);
            // hierarchy_data.all_descendants.forEach(d => d.children = []);

            state.hierarchy_data = hierarchy_data;
            state.in_length_compare = false;
            console.log(hierarchy_data)
        },
        set_class_info(state, class_info){
            console.log("set class info");
            state.class_info = class_info;
            class_info.forEach(d => {
                d.num = d.labeled_num;
                d.id = parseInt(d.name);
                d.full_name = state.class_names[parseInt(d.name)];
                d.entropy = Math.pow(d.entropy, 0.7);
            });
            let selected_idx =  [9, 10, 11, 12, 13, 14, 17];
            class_info = class_info.filter(d => selected_idx.indexOf(d.id) > -1);
            state.class_info = class_info;
        },
        set_selected_frames(state, frames){
            console.log("set images");
            state.selected_frames = frames;
        },
        set_selected_details(state, data){
            console.log("set_selected_details");
            state.selected_details = data;
        },
        set_selected_class(state, selected_class){
            console.log("set_selected_class");
            state.selected_class = selected_class;
        },
        set_selected_cluster(state, selected_cluster){
            console.log("set_selected_cluster");
            state.selected_cluster = selected_cluster;
        },
        set_selected_ids(state, selected_ids){
            console.log("set_selected_ids");
            state.selected_ids = selected_ids;
        },
        set_expand_node_ids(state, change){
            console.log("set_expand_node_id", change);
            if (change.mode === "single"){
                let id = change.data;
                if (state.expand_node_ids.indexOf(id) > -1){
                    let idx = state.expand_node_ids.indexOf(id);
                    state.expand_node_ids.splice(idx, 1);
                }
                else state.expand_node_ids.push(id);
                if (state.expand_node_ids[0] === "clean") state.expand_node_ids[0] = "cleaned";
            }
            else if (change.mode === "full"){
                state.expand_node_ids = change.data;
            }
            else{
                alert("set expand node ids error");
            }
        },
        set_selected_action(state, selected_action){
            // console.log("set_selected_action");
            state.selected_action = selected_action;
        },
        set_selected_video(state, selected_video){
            // console.log("set_selected_video");
            state.selected_video = selected_video;
        },
        set_key_frames(state, key_frames){
            // console.log("set_key_frames");
            state.key_frames = key_frames.frames;
        },
        set_anchor_alignments(state, anchor_alignments){
            let y_layout = anchor_alignments.y_layout
            let root = {
                data: anchor_alignments,
                all_children: [],
                _total_width: 0,
                aligns: anchor_alignments.aligns,
                single_frame: anchor_alignments.single_frame,
                y_layout,
                action_list: anchor_alignments.action_list,
                action_id: anchor_alignments.action_id,
                is_anchor_alignment: 1,
                api: 0,
            }
            anchor_alignments.actions.forEach((action, siblings_id) => {
                let parent = root
                let e = {
                    data: action,
                    siblings_id,
                    parent,
                    mat: parent.aligns[siblings_id].filter(d => d >= 0).map(d => [d]),
                }
                e.actions = [e.data.name];
                e.rep_action = e.data.name; // a leaf node itself is its representative action
    
                e.vid = e.data.video_id;
                e.start_idx = e.data.bound[0];
                e.end_idx = e.data.bound[1];

                let all_frames = [];
                for (let i = e.start_idx; i <= e.end_idx; i++){
                    all_frames.push(i);
                }
                e.all_frames = all_frames;
                e.rep_frames = all_frames; //e.data.key_frame.slice(0,3); 
                e.rep_frames = e.rep_frames.map(d => e.vid + "-" + d);
                e.all_frames = e.all_frames.map(d => e.vid + "-" + d);

                e.single_frame = e.data.single_frame - e.data.bound[0];
                e.labels = new Array(e.mat.length).fill([-1]);
                e.labels[e.data.single_frame - e.data.bound[0]] = [1];
                e.y_layout_height = new Array(e.mat.length).fill(1.0 / 10);
                root.all_children.push(e)
            })
            root.children = root.all_children 
            state.anchor_alignments = root;
        },
        set_constraint(state, constraint){
            state.constraint = constraint;
        },
        set_history(state, history){
            state.history = history;
            console.log("history", state.history)
        },
        push_history(state, step){
            state.history.push(step);
            console.log("history", state.history)
        },
        pop_history(state, target){
            state.history = state.history.slice(0, target);
            console.log("history", state.history)
        },
        set_work_history(state, work_history){
            state.work_history = work_history;
        },
        compare_action_length(state, dict){
            state.hierarchy_data.eachAfter(e => {
                if (!e.all_children){
                    e.compare_lengths = [dict[e.id]]
                }
                else{
                    e.compare_lengths = e.all_children.map(d => d.compare_lengths).reduce((p,v) => p.concat(v), []);
                }
            });
            state.in_length_compare = true;
        },
        set_in_length_compare(state, value){
            state.in_length_compare = value;
        },
        set_compare_step(state, step){
            state.compare_step = step;
        }
    },
    actions:{ // function to fetch data from backend
        async fetch_manifest({commit, state}, key){
            console.log("fetch_manifest");
            const resp = await axios.post(`${state.server_url}/Corpus/GetManifest`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            console.log("get manifest", resp);
            commit("set_manifest_data", JSON.parse(JSON.stringify(resp.data)));
        },
        async fetch_hierarchy_meta_data({commit, state}, key){
            console.log("fetch_hierarchy_meta_daata");
            const resp = await axios.post(`${state.server_url}/Corpus/GetHiearchyMetaData`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            console.log("get hierarchy meta data", resp);
            commit("set_class_info", JSON.parse(JSON.stringify(resp.data)));

        },
        async fetch_hierarchy({commit, state}, key){
            console.log("fetch_hierarchy");
            const resp = await axios.post(`${state.server_url}/Corpus/GetHierarchy`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            console.log("get hierarchy", resp);
            commit("set_hierarchy", JSON.parse(JSON.stringify(resp.data)));
        },
        async fetch_video_info({commit, state}){
            console.log("fetch_video_info");
            let key = {"id": parseInt(state.selected_action.id.split("-")[0]),
                    "class": state.selected_class}
            const resp = await axios.post(`${state.server_url}/Video/GetVideoInfo`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            console.log("get_video_info", resp);
            commit("set_selected_video", JSON.parse(JSON.stringify(resp.data)));
        },
        async fetch_key_frames({commit, state}, key){
            // console.log("fetch_key_frames");
            const resp = await axios.post(`${state.server_url}/Video/GetRepFrames`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            // console.log("get_key_frames", resp);
            commit("set_key_frames", JSON.parse(JSON.stringify(resp.data)));
        },
        // async fetch_
        async fetch_key_frames_directly({state}, key){
            const resp = await axios.post(`${state.server_url}/Video/GetActionFrames`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            return JSON.parse(JSON.stringify(resp.data));
        },

        async fetch_anchor_alignments({state, commit}, key){
            const resp = await axios.post(`${state.server_url}/Video/GetAlignmentOfAnchorAction`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            commit("set_anchor_alignments", JSON.parse(JSON.stringify(resp.data)));
        },

        async set_corpus_arguments({state}, key){
            const resp = await axios.post(`${state.server_url}/Corpus/SetCorpusArgs`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            console.log("set corpus args", resp, key);
        },

        async fetch_pred_scores_of_video_with_given_boundary({state}, key){
            const resp = await axios.post(`${state.server_url}/Video/GetPredScoresOfVideoBoundary`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            return JSON.parse(JSON.stringify(resp.data))
        },

        async fetch_update({dispatch, commit, state}){
            let key = {constraint: state.constraint};
            const resp = await axios.post(`${state.server_url}/Corpus/Update`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            if(resp.data.msg === "ok"){
                await dispatch("fetch_hierarchy_meta_data");
                await dispatch("fetch_hierarchy", {id: state.selected_class});
            }
        }, 

        async remove_action({dispatch, commit, state}){
            let key = {action_id: state.history[state.history.length - 1].id};
            const resp = await axios.post(`${state.server_url}/Corpus/RemoveAction`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            if(resp.data.msg === "ok"){
                state.history.pop();
                await dispatch("fetch_hierarchy_meta_data");
                await dispatch("fetch_hierarchy", {id: state.selected_class});
            }
        }, 
        
        async fetch_recommendation({state}, label_info) {
            let key = {
                class: state.selected_class,
                action_id: label_info[0],
                labeled_frame_id: label_info[1],
                bound_pos: label_info[2]
            };
            const resp = await axios.post(`${state.server_url}/Corpus/GetRecommendation`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            return JSON.parse(JSON.stringify(resp.data))
        },

        async fetch_work_history({commit, state}){
            let key = {};
            const resp = await axios.post(`${state.server_url}/Corpus/GetHistory`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            commit("set_work_history", JSON.parse(JSON.stringify(resp.data)));
        }, 

        async fetch_step_history({commit, state}, step){
            let key = {class: state.selected_class, step: step};
            const resp = await axios.post(`${state.server_url}/Corpus/GetStepHistory`, key, 
                {headers: {
                    "Content-Type":"application/json",
                    "Access-Control-Allow-Origin": "*",
                }});
            commit("compare_action_length", JSON.parse(JSON.stringify(resp.data)));
        }, 
    },
    modules:{
        // empty
    }
})

export default store