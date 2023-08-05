import * as d3 from 'd3';
import * as Global from './global';
import * as d3ContextMenu from "d3-context-menu";
import { AlignmentLayout } from "./layout_alignment";
import { interpolatePath  } from "d3-interpolate-path" // ES6
import { transition } from 'd3';

const pathInterpolate = function(start, end) {
    // console.log('interpolate')
    // console.log(start)
    // console.log(end)
    return interpolatePath(start, end)
}

const omg = 1 / 8;
const beta = 1 / 6;
const use_inside_line = false;

const AlignMentsRender = function (parent) {
    let that = this;
    that.parent = parent;
    that.space = 20;
    that.button_width = 28;
    that.button_height = 14;
    that.frame_width = 110;
    that.frame_height = 110;
    that.ratio = [0.0, 0.9];
    that.stack = [];
    that.operation = {
        'must-link': [],
        'must-not-link': [],
        'adjusted-boundaries': [-1, -1],
    };
    that.bread_width = 25;
    that.bread_space = 15;
    that.text_size = 14;
    that.drag_min_dist = 15
    that.ptr_width = 25;
    that.ptr_y_margin = 58;
    that.last_action = null;
    that.video = {
        width: 400,
        view_id: "selected-action-video",
        view: null,
        id: null,
        action: null,
        stride: 16,
        fps: 25,
        fid_to_t: fid => (fid * that.video.stride 
            + that.video.stride / 2) / that.video.fps,
        delta: 16 / 25 * 1000,
    }
    that.axis_margin = 10;
    that.click_timer = null;

    that.alignment_layout = new AlignmentLayout(that);

    that.update_info_from_parent = function () {
        // canvas
        that.aligns_group = that.parent.aligns_group;

        // layout
        that.layout_width = that.parent.aligns_width;
        that.layout_height = that.parent.aligns_height;

        // animation
        that.create_ani = that.parent.create_ani;
        that.update_ani = that.parent.update_ani;
        that.remove_ani = that.parent.remove_ani;

        // video
        if(that.video.view === null){
            that.video.view = document.getElementById(that.video.view_id);
        }
        that.video.view.style.visibility = "hidden";
    }

    that.update_info_from_parent();

    that.sub_component_update = function () {
        // update info
        that.update_info_from_parent();

        // update state
        that.data = that.parent.hierarchy_data;
        that.selected_cluster = that.parent.selected_cluster;
        that.anchor_alignments = that.parent.anchor_alignments;
        that.in_length_compare = that.parent.in_length_compare;

        that.stack = [];
        that.stack.push(that.data);

        // reback history state
        let finish = false;
        that.return_for_answer = false;
        that.parent.history.forEach(d => {
            if(finish) return;
            if(!d.is_comparison){
                if(d.id >= that.data.all_children.length){
                    finish = true;
                    return;
                }
                that.data = that.data.all_children[d.id];
                that.stack.push(that.data);
            }
            else{
                that.parent.fetch_anchor_alignments({
                    class: that.parent.selected_class,
                    id: d.id
                })
                that.return_for_answer = true;
            }
        })
        // update view
        if(that.return_for_answer)
            return;
        that.sub_component_view_update();
    }

    that.sub_component_view_update = function () {
        // update cards view
        that.cards = that.alignment_layout.update_layout(that.data);
        // console.log("cards", that.cards)

        // that.e_cards = that.aligns_group.selectAll(".card-group")
        //     .data(that.cards, d => d.id);
        // let cluster_groups = that.cards_create(that.e_cards);
        // that.cards_update(that.e_cards);
        // that.cards_remove(that.e_cards);
        // that.e_clusters = that.aligns_group.selectAll(".cluster-group")
        //     .data(that.cards[0].clusters);
        // let merged_clusters = that.e_clusters;
        // if(!that.cards[0].normal_layout){
        //     that.cards[0].current_action = that.cards[0].actions[0];
        //     that.cards[0].current_action.selected = true;
        // }
        that.sub_component_view_refresh()
    }

    that.sub_component_view_refresh = function () {
        let card = that.cards[0];

        that.lower_background_update(that.aligns_group, card);
        that.cluster_subcomponent_update(that.aligns_group, card.clusters);
        that.action_subcomponent_update(that.aligns_group, card.actions);
        that.selected_action_subcomponent_update(that.aligns_group, card)
        that.frame_band_subcomponent_update(that.aligns_group, card.rep_frames);
        that.frame_point_subcomponent_update(that.aligns_group, card.clusters);
        that.upper_background_update(that.aligns_group, card);

        // update breadcrumbs
        that.breadcrumbs_layout();
        that.e_breads = that.aligns_group.selectAll(".breadcrumb-group")
            .data(that.breadcrumbs, d => d.index);
        that.breadcrumbs_create(that.e_breads);
        that.breadcrumbs_update(that.e_breads);
        that.breadcrumbs_remove(that.e_breads);

        that.last_card = card
    }

    that.draw_bound = function (d) {
        let x1 = (that.ratio[0] * d.width - 2 * that.space - 3 * that.frame_width) / 3 - 0.6 * that.space,
            x2 = that.ratio[0] * d.width - 2 * that.space,
            x3 = that.ratio[0] * d.width + that.space,
            x4 = x2 + that.ratio[1] * d.width + that.space,
            x5 = (x2 + x3) / 2;
        let y1 = 0,
            // y2 = (d.height - that.frame_height) / 2 - 0.6 * that.space,
            y2 = d.rep_frame_center_y - that.frame_height / 2 - 0.6 * that.space,
            // y3 = d.height - y2,
            y3 = d.rep_frame_center_y + that.frame_height / 2 + 0.6 * that.space,
            y4 = d.height,
            // y4 = that.frame_height,
            y5 = (y1 + y2) / 2,
            y6 = (y3 + y4) / 2;
        let cx1 = x2 + 0.6 * that.space,
            cx2 = x3 - 0.6 * that.space,
            cdy = 3;

        // return `M ${x1},${y2}` + `L ${x2},${y2}` + `Q ${cx1},${y2 - cdy} ${x5},${y5}`
        //     + `Q ${cx2},${y1 + cdy} ${x3},${y1}` + `L ${x4},${y1}` + `L ${x4},${y4}` + `L ${x3},${y4}`
        //     + `Q ${cx2},${y4 - cdy} ${x5},${y6}` + `Q ${cx1},${y3 + cdy} ${x2},${y3}` + `L ${x1},${y3}` + `L ${x1},${y2}`;
        return `M ${x3},${y1}` + `L ${x4},${y1}` + `L ${x4},${y4}` + `L ${x3},${y4}z`;
    }

    that.stroke_color = function(d) {
        if(d.selected) return Global.Orange;
        if(d.is_anchor) return Global.Purple //"rgba(255, 169, 83, 0.8)";
        if(d.action_id[0] === 'N') return Global.MidBlue;
        // if(d.action_id[0] === 'N' && d.type === 'cluster') return Global.MidBlue;
        return Global.BlueGray;
    }

    that.cards_create = function (elem) {
        let create = elem.enter()
            .append("g")
            .attr("class", "card-group")
            .attr("id", d => "cg-" + d.id)
            .attr("transform", d => `translate(${d.x}, ${d.y})`);
        create
            .style("opacity", 0)
            .transition()
            .duration(that.create_ani)
            .delay(that.update_ani + that.remove_ani)
            .style("opacity", 1);
        return create;
    }
    that.cards_update = function (elem) {
        elem.transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("transform", d => `translate(${d.x}, ${d.y})`);
    }
    that.cards_remove = function (elem) {
        elem.exit()
            .transition()
            .duration(that.remove_ani)
            .style("opacity", 0)
            .remove();
    }

    that.recommend_update = function (resp) {
        console.log("recommend", resp)
        if(resp.msg !== "ok") return;
        
        let action = that.cards[0].current_action;
        // move boundary
        if(resp.recom_direct === "left"){
            action.bound[0] = Math.min(action.bound[0], resp.recom_pos - 2);
            action.initial_bound[0] = resp.label_id;
        }
        else{
            action.bound[1] = Math.max(action.bound[1], resp.recom_pos + 2);
            action.initial_bound[1] = resp.label_id;
        }
        action.recom_direct = resp.recom_direct;
        action.recom_pos = resp.recom_pos;
        that.selected_action_subcomponent_update(that.aligns_group, that.cards[0]);
        d3.select(`#selected-action-frame-${action.recom_pos}`).raise()
    }

    that.selected_action_subcomponent_update = function (groups, d) {
        let data = []

        const frame_height = that.alignment_layout.large_frame_height
        const frame_width = that.alignment_layout.large_frame_width
        const margin = frame_width
        const width = d.width - margin * 2 - that.video.width;
        let action = d.current_action
        if (action !== null) {
            const rep_frames = new Set(action.rep_frames)
            for (let frame_id = action.bound[0]; frame_id <= action.bound[1]; ++frame_id) {
                const id = `${action.vid}-${frame_id}`
                let pos = 0
                if(frame_id < action.single_frame) pos = -1;
                else if(frame_id > action.single_frame) pos = 1;
                data.push({
                    y: d.line_height + 130,
                    rep: rep_frames.has(id),
                    id: id,
                    frame_id: frame_id,
                    is_initial: frame_id >= action.initial_bound[0] && frame_id <= action.initial_bound[1],
                    pos: pos
                })
            }
            const image_margin = Math.min(frame_width + 5, (width - frame_width) / data.length)
            const start_x = (width - data.length * image_margin - frame_width) / 2
            for (let i = 0; i < data.length; ++i) {
                data[i].x = i * image_margin + start_x;
                data[i].image_margin = image_margin
            }

            that.selected_action_data = data;
            that.selected_width = frame_width;
            that.operation['adjusted-boundaries'] = [action.initial_bound[0], action.initial_bound[1]]
            //<path d="M731.0336 859.8528V164.1472c0-40.1408-48.5376-60.3136-77.0048-31.8464L306.176 480.1536c-17.6128 17.6128-17.6128 46.1824 0 63.7952l347.8528 347.8528c28.4672 28.3648 77.0048 8.2944 77.0048-31.9488z" p-id="1603" data-spm-anchor-id="a313x.7781069.0.i0"></path>
            //<path d="M292.9664 164.1472v695.808c0 40.1408 48.5376 60.3136 77.0048 31.8464L717.824 543.8464c17.6128-17.6128 17.6128-46.1824 0-63.7952L369.9712 132.1984c-28.4672-28.3648-77.0048-8.2944-77.0048 31.9488z" p-id="1810"></path>

        }

        let frames = groups
            .selectAll(".selected-action-frames-group")
            .data(data, d => d.id)

        let g = frames.enter()
            .append('g')
            .attr('transform', d => `translate(${d.x + margin},${d.y}) scale(1)`)
            .attr('class', 'selected-action-frames-group')
            .attr('id', d => `selected-action-frame-${d.frame_id}`)
            .style('opacity', 0)
        
        g.transition()
            .duration(this.create_ani)
            .style('opacity', 1)

        g.append('text')
            .attr("dx", d => d.image_margin / 2)
            .attr("dy", -20)
            .attr("font-weight", 650)
            .attr("text-anchor", "middle")
            .attr("fill", Global.DeepGray)
            .text(d => `${d.frame_id}`)
            // .text(d => d.pos !== 0 ? `${d.frame_id}`: `⭐${d.frame_id}`)

        g.filter(d => d.pos === 0)
            .append("path")
            .attr("class", "anchor-star")
            .attr("d", "M14.1,0.43l3.44,8.05l8.72,0.78c0.39,0.03,0.67,0.37,0.64,0.76c-0.02,0.19-0.1,0.35-0.24,0.47l0,0 l-6.6,5.76l1.95,8.54c0.09,0.38-0.15,0.75-0.53,0.84c-0.19,0.04-0.39,0-0.54-0.1l-7.5-4.48l-7.52,4.5 c-0.33,0.2-0.76,0.09-0.96-0.24c-0.1-0.16-0.12-0.35-0.08-0.52h0l1.95-8.54l-6.6-5.76c-0.29-0.25-0.32-0.7-0.07-0.99 C0.3,9.35,0.48,9.28,0.66,9.27l8.7-0.78l3.44-8.06c0.15-0.36,0.56-0.52,0.92-0.37C13.9,0.13,14.03,0.27,14.1,0.43L14.1,0.43 L14.1,0.43z")
            .attr("transform", d => `translate(${d.image_margin / 2 - 7}, ${-20}) scale(0.55)`)
            .style("fill", Global.Orange)
            .style("stroke", "white")
            .style("stroke-width", 3)
            .style("opacity", 1)
        let img_margin = 5;
        g.append("rect")
            .attr("width", frame_width)
            .attr("height", frame_height)
            .attr("fill", "white")
            .style("pointer-event", "none");
        g.append('image')
            .attr("x", img_margin)
            .attr("y", img_margin)
            .attr('width', frame_width - img_margin * 2)
            .attr('height', frame_height - img_margin * 2)
            .attr('xlink:href', d => `${that.parent.server_url}/Frame/single_frame?filename=${d.id}`)
            .on("click", function(ev, d) {
                that.click_timer && clearTimeout(that.click_timer);
                that.click_timer = setTimeout(() => {
                    that.play_video_from_fid(d.frame_id);
                }, 150)
            })
            .on("dblclick", function(ev, d) {
                that.click_timer && clearTimeout(that.click_timer);
                that.pause_video_to_fid(d.frame_id);
            })
            .on("mouseenter", function(ev, d) {
                that.in_show_image = true;
                d3.select(`#selected-action-frame-${d.frame_id - 1}`).raise()
                d3.select(`#selected-action-frame-${d.frame_id + 1}`).raise()
                d3.select(`#selected-action-frame-${d.frame_id}`).raise()
                d3.select(this.parentNode)
                    .transition()
                    .duration(Global.QuickAnimation)
                    .attr('transform', d => `translate(${d.x + margin - 0.5 * frame_width},${d.y - 0.5 * frame_height}) scale(1.5)`)
                    .style('opacity', 1)
            })
            .on("mouseout", function(ev, d) {
                // d3.select(`#selected-action-frame-${d.frame_id}`).order()
                for (let i = 0; i < data.length; i++){
                    d3.select(`#selected-action-frame-${data[i].frame_id}`).raise()
                }
                console.log("mouseout", d3.selectAll(".selected-action-frames-group"))
                // d3.select(this.parentNode).order()
                // d3.select(this.parentNode).interrupt()
                d3.select(this.parentNode)
                    .transition()
                    .duration(Global.QuickAnimation)
                    .attr('transform', d => `translate(${d.x + margin},${d.y}) scale(1)`)
                    .style('opacity', 1)
                that.in_show_image = false;
                setTimeout(function(){
                    if(!that.in_show_image){
                        d3.select('.selected-action-frame-info').raise()
                    }
                }, Global.QuickAnimation);
            })
            .on("contextmenu", d3ContextMenu(data => {
                let choice_list = [];
                choice_list.push({
                    title: "Left-Boundary",
                    action: function(d) {
                        console.log("Boundary left!", action, d);
                        that.parent.last_boundary_id = d.id;
                        that.parent.bound_pos = -1;
                        action.initial_bound[0] = d.frame_id;
                        action.recom_pos = -1;
                        that.operation['adjusted-boundaries'][0] = action.initial_bound[0]
                        that.selected_action_subcomponent_update(that.aligns_group, that.cards[0]);
                    },
                    disabled: data.pos > 0
                });
                choice_list.push({
                    title: "Right-Boundary",
                    action: function(d) {
                        console.log("Boundary right!", action, d);
                        that.parent.last_boundary_id = d.id;
                        that.parent.bound_pos = 1;
                        action.initial_bound[1] = d.frame_id;
                        action.recom_pos = -1;
                        that.operation['adjusted-boundaries'][1] = action.initial_bound[1]
                        that.selected_action_subcomponent_update(that.aligns_group, that.cards[0]);
                    },
                    disabled: data.pos < 0
                });
                return choice_list;
            }));

        frames.transition()
            .duration(this.update_ani)
            .attr('transform', d => `translate(${d.x + margin},${d.y})`)
        frames.select("text")
            .transition()
            .duration(this.update_ani) 
            .attr("dx", d => d.image_margin / 2)
            .attr("dy", -20)
        frames.select("path.anchor-star")
            .transition()
            .duration(this.update_ani) 
            .attr("transform", d => `translate(${d.image_margin / 2 - 7}, ${-20}) scale(0.55)`)

        frames.exit()
            .attr('opacity', 0)
            .remove();

        groups
            .selectAll(".selected-action-frame-area")
            .remove()

        if (d.current_action != null) {
            let left_x = Math.min(...data.map(d => d.x))
            let right_x = Math.max(...data.map(d => d.x)) + frame_width * 2
            let center_x = data.filter(d => d.pos === 0)[0].x
            let y = data[0].y
            let left_initial_x = Math.min(...data.filter(d => d.is_initial).map(d => d.x)) + frame_width
            let right_initial_x = Math.max(...data.filter(d => d.is_initial).map(d => d.x)) + frame_width * 2
            let recom = data.filter(d => d.frame_id === action.recom_pos);
            // let recom_x = (recom.length > 0) ? recom[0].x + frame_width * 1.5: -1;
            let recom_x = (recom.length > 0) ? recom[0].x + + frame_width + recom[0].image_margin * 0.5: -1;


            let yScale = d3.scaleLinear()
                .domain([0, 1])
                .range([that.cards[0].single_area_height, 0])

            let line = d3.line()
                .x(d => d.x)
                .y(d => d.y);

            let line2 = d3.line()
                .curve(d3.curveCardinal)
                .x(d => d.x)
                .y(d => d.y);

            groups.selectAll("g.recommend-ptr")
                .data(["ptr"])
                .enter()
                .append("g")
                .attr("class", "recommend-ptr")
                .style("opacity", 0)
                .attr("transform", `translate(${recom_x - that.ptr_width / 2}, ${y - that.ptr_y_margin})`)
                .append("svg")
                .attr("width", that.ptr_width)
                .attr("height", that.ptr_width)
                .attr("viewBox", "0 0  1024 1024")
                .append("path")
                .attr("d", "M564.224 44.032q43.008 0 58.368 20.48t15.36 65.536q0 20.48 0.512 64.512t0.512 93.696 0.512 96.768 0.512 74.752q0 38.912 7.68 61.952t35.328 22.016q19.456 0 48.128 1.024t49.152 1.024q35.84 0 45.568 18.944t-13.824 49.664q-24.576 30.72-57.344 72.704t-68.096 86.016-69.12 86.528-59.392 75.264q-23.552 29.696-45.568 30.72t-45.568-27.648q-24.576-29.696-57.344-69.632t-67.072-82.432-67.584-83.968-59.904-74.24q-29.696-35.84-22.528-58.88t44.032-23.04l24.576 0q14.336 0 29.696-0.512t30.208-1.536 26.112-1.024q26.624 0 32.768-15.36t6.144-41.984q0-29.696-0.512-77.824t-0.512-100.352-0.512-101.376-0.512-79.872q0-13.312 2.048-27.648t9.728-26.112 20.992-19.456 36.864-7.68q27.648 0 53.248-0.512t57.344-0.512z")
                .attr("fill", "#f03b21");

            that.parent.fetch_pred_scores_of_video_with_given_boundary({
                "id": action.vid,
                "class": that.parent.selected_class,
                "bound": action.bound,
            }).then((res) => {
                let points = data.map((d, i) => ({
                    x: d.x + frame_width * 1.5,
                    y: yScale(res.scores[i]),
                }))
                points.splice(0, 0, { x: points[0].x - frame_width * 0.5, y: points[0].y })
                points.push({
                    x: points[points.length - 1].x + frame_width * 0.5,
                    y: points[points.length - 1].y
                })
                let path1 = `M${points[0].x} ${yScale(0)} L${line(points).slice(1)} L${points[points.length - 1].x} ${yScale(0)} Z`
                let path2 = line2(points)

                // groups
                //     .append("path")
                //     .attr("transform", `translate(0,${y + frame_height + that.alignment_layout.margin})`)
                //     .attr("class", "selected-action-frame-area")
                //     .attr("d", path2)
                //     .style("fill", "none")
                //     .style("stroke", Global.DarkBlueGray)
                //     .style("stroke-width", "3px")
                //     .style("opacity", 0)
                //     .transition()
                //     .duration(that.create_ani)
                //     .delay(that.update_ani)
                //     .style("opacity", 1)

                // groups
                //     .append("path")
                //     .attr("transform", `translate(0,${y + frame_height + that.alignment_layout.margin})`)
                //     .attr("class", "selected-action-frame-area")
                //     .attr("d", path1)
                //     .style("fill", Global.Blue40)
                //     .style("fill-opacity", .5)
                //     .style("stroke", "none")
                //     .style("opacity", 0)
                //     .transition()
                //     .duration(that.create_ani)
                //     .delay(that.update_ani)
                //     .style("opacity", 1)

                // groups.append("text")
                //     .attr("class", "selected-action-frame-area")
                //     .attr("transform", `translate(${left_x + frame_width / 4},${y + frame_height * 1.25 + that.alignment_layout.single_area_height / 2})`)
                //     .attr("font-size", "16px")
                //     .text("Confidence")
                
                // let y_control_line = y + frame_height * 1.25+ that.alignment_layout.single_area_height + 4;
                // groups.append("line")
                //     .attr("class", "selected-action-frame-area")
                //     .attr("id", "video-control-line")
                //     .attr("x1", left_x + frame_width - that.axis_margin)
                //     .attr("y1", y_control_line)
                //     .attr("x2", right_x + that.axis_margin)
                //     .attr("y2", y_control_line)
                //     .attr("stroke", Global.DeepGray)
                //     .attr("stroke-width", 4)
                //     .on("click", function(ev) {
                //         that.click_timer && clearTimeout(that.click_timer);
                //         that.click_timer = setTimeout(() => {
                //             let min_index = -1, min_dist = 10000;
                //             let delta_x = document.getElementById("video-control-line").getBoundingClientRect().left 
                //                 + document.documentElement.scrollLeft - (left_x + frame_width - that.axis_margin);
                //             that.selected_action_data.forEach((d, index) => {
                //                 let rel_ev_x = ev.x - delta_x;
                //                 let rel_d_x = d.x + frame_width * 3 / 2;
                //                 let dist = Math.abs(rel_d_x - rel_ev_x);
                //                 if(dist < min_dist){
                //                     min_dist = dist;
                //                     min_index = index;
                //                 }
                //             });
                //             let frame_id = that.selected_action_data[min_index].frame_id
                //             that.play_video_from_fid(frame_id);
                //         }, 150)
                //     })
                //     .on("dblclick", function(ev, d) {
                //         that.click_timer && clearTimeout(that.click_timer);
                //         let min_index = -1, min_dist = 10000;
                //         let delta_x = document.getElementById("video-control-line").getBoundingClientRect().left 
                //             + document.documentElement.scrollLeft - (left_x + frame_width - that.axis_margin);
                //         that.selected_action_data.forEach((d, index) => {
                //             let rel_ev_x = ev.x - delta_x;
                //             let rel_d_x = d.x + frame_width * 3 / 2;
                //             let dist = Math.abs(rel_d_x - rel_ev_x);
                //             if(dist < min_dist){
                //                 min_dist = dist;
                //                 min_index = index;
                //             }
                //         });
                //         let frame_id = that.selected_action_data[min_index].frame_id
                //         that.pause_video_to_fid(frame_id);
                //     })
                //     .style("opacity", 0)
                //     .transition()
                //     .duration(that.create_ani)
                //     .delay(that.update_ani)
                //     .style("opacity", 0.5)

                // groups.append("line")
                //     .attr("class", "selected-action-frame-area")
                //     .attr("id", "selected-action-video-pos")
                //     .attr("x1", center_x + frame_width * 3 / 2)
                //     .attr("y1", y_control_line - 4)
                //     .attr("x2", center_x + frame_width * 3 / 2)
                //     .attr("y2", y_control_line + 4)
                //     .attr("stroke", Global.DarkBlueGray)
                //     .attr("stroke-width", 6)
                //     .style("opacity", 0)
                //     .transition()
                //     .duration(that.create_ani)
                //     .delay(that.update_ani)
                //     .style("opacity", 1);
            
                groups.selectAll("g.recommend-ptr")
                    .transition()
                    .duration(that.remove_ani)
                    .style("opacity", 0);

                if(recom_x > -1){
                    groups.selectAll("g.recommend-ptr")
                        .attr("transform", `translate(${recom_x - that.ptr_width / 2}, ${y - that.ptr_y_margin})`)
                        .transition()
                        .duration(that.create_ani)
                        .delay(that.create_ani + that.update_ani - that.remove_ani)
                        .style("opacity", 1)
                }
                    
            })

            let info_elem = groups.selectAll(".selected-action-frame-info")
                .data([null])
        
            let info = info_elem.enter().append('g')
                .attr("class", "selected-action-frame-info")

            info.append("rect")
                .attr("class", "bound")
                .attr('x', left_initial_x - 3)
                .attr('y', y - 4)
                .attr('width', right_initial_x - left_initial_x + 6)
                .attr('height', frame_height + 7)
                .attr('fill', 'none')
                .attr('stroke-width', 5)
                .style('stroke', Global.Orange)
                .style("opacity", 0)
                .transition()
                .duration(that.create_ani)
                .delay(that.update_ani)
                .style("opacity", 1)

            info.append("path")
                .attr("class", "left-button")
                .attr("transform", `translate(${left_x + frame_width / 4},${y + frame_height / 4}) scale(0.075)`)
                .attr("d", "M731.0336 859.8528V164.1472c0-40.1408-48.5376-60.3136-77.0048-31.8464L306.176 480.1536c-17.6128 17.6128-17.6128 46.1824 0 63.7952l347.8528 347.8528c28.4672 28.3648 77.0048 8.2944 77.0048-31.9488z")
                .style("fill", Global.GrayColor)
                .style("stroke", "none")
                .style("opacity", .5)
                .on("mouseenter", function () {
                    d3.select(this)
                        .style("opacity", 1)
                })
                .on("mouseout", function () {
                    d3.select(this)
                        .style("opacity", .5)
                })
                .on("click", function () {
                    d.current_action.bound[0] -= 3
                    d.current_action.recom_pos = -1
                    that.sub_component_view_refresh()
                })

            info.append("path")
                .attr("class", "right-button")
                .attr("transform", `translate(${right_x + frame_width / 4},${y + frame_height / 4}) scale(0.075)`)
                .attr("d", "M292.9664 164.1472v695.808c0 40.1408 48.5376 60.3136 77.0048 31.8464L717.824 543.8464c17.6128-17.6128 17.6128-46.1824 0-63.7952L369.9712 132.1984c-28.4672-28.3648-77.0048-8.2944-77.0048 31.9488z")
                .style("fill", Global.GrayColor)
                .style("stroke", "none")
                .style("opacity", .5)
                .on("mouseenter", function () {
                    d3.select(this)
                        .style("opacity", 1)
                })
                .on("mouseout", function () {
                    d3.select(this)
                        .style("opacity", .5)
                })
                .on("click", function () {
                    d.current_action.bound[1] += 3
                    d.current_action.recom_pos = -1
                    that.sub_component_view_refresh()
                })

            info_elem.raise()

            info_elem.select(".bound")
                .transition()
                .duration(that.update_ani)
                .attr('x', left_initial_x - 3)
                .attr('width', right_initial_x - left_initial_x + 6)
                .attr('height', frame_height + 7)

            info_elem.select(".left-button")
                .transition()
                .duration(that.update_ani)
                .attr("transform", `translate(${left_x + frame_width / 4},${y + frame_height / 4}) scale(0.075)`)

            info_elem.select(".right-button")
                .transition()
                .duration(that.update_ani)
                .attr("transform", `translate(${right_x + frame_width / 4},${y + frame_height / 4}) scale(0.075)`)

            groups.append("line")
                .attr("class", "selected-action-frame-area")
                .attr('x1', 0)
                .attr('x2', d.width)
                .attr('y1', y - 100)
                .attr('y2', y - 100)
                .style('stroke', Global.DeepGray)
                .style('opacity', .4)
                .style('stroke-width', 1.5)
            
            if(that.video.id !== d.current_action.vid){
                that.video.id = d.current_action.vid;
                that.video.view.src = `${that.parent.server_url}/Video/GetVideo?id=${that.video.id}`;
            }
            if(that.video.action !== d.current_action.single_frame){
                that.video.action = d.current_action.single_frame;
                that.video.view.currentTime = that.video.fid_to_t(d.current_action.single_frame);
            }
            that.video.view.style.visibility = "visible";
        } 
        else {
            groups.selectAll(".selected-action-frame-info")
                .style('opacity', 1)
                .transition()
                .duration(that.remove_ani)
                .style('opacity', 0)
                .remove()

            groups.selectAll("g.recommend-ptr")
                .style("opacity", 0);

            that.video.view.style.visibility = "hidden";
        }
    }

    that.action_subcomponent_update = function (groups, data) {
        let elem = groups
            .selectAll(".action-group")
            .data(data, d => d.action_id);
        let action_groups = that.action_line_create(elem, groups);
        that.action_line_update(elem, groups);
        that.action_line_remove(elem, groups);
        let merged_action_groups = elem.merge(action_groups);
        that.merged_action_groups = merged_action_groups
    };

    that.action_line_create = function (elem, groups) {
        let create = elem.enter()
            .append("g")
            .attr("class", "action-group")
            .attr("id", d => "ag-" + d.action_id)
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
            .style("opacity", 0);

        create
            .transition()
            .duration(that.create_ani)
            .delay(that.update_ani + that.remove_ani + that.create_ani)
            .style("opacity", d => d.type == 'fake' ? 0 : 1)

        let opa = 0.6;
        create
            .append("path")
            .attr("class", "action-start-line")
            .attr("d", d => d.start_line)
            // .style("stroke-dasharray", "4 4 8 4 12 4")
            .style("fill", "none")
            .style("stroke-width", d => d.width)
            .style("stroke", d => that.stroke_color(d))
            .style("opacity", d => d.type === "cluster" ? opa: 1)

        create
            .append("path")
            .attr("class", "action-center-line")
            .style("stroke-width", d => d.width)
            .style("stroke", d => that.stroke_color(d))
            .style("fill", "none")
            .attr("d", d => d.center_line)
            .attr("d2", d => d.center_line_z)
            .style("opacity", d => d.type === "cluster" ? opa: 1)

        create
            .append("path")
            .attr("class", "action-end-line")
            .style("stroke-width", d => d.width)
            // .style("stroke-dasharray", "4 4 8 4 12 4")
            .style("stroke", d => that.stroke_color(d))
            .style("fill", "none")
            .attr("d", d => d.end_line)
            .style("opacity", d => d.type === "cluster" ? opa: 1)
        
        // inside lines
        let clusters = create.filter(d => d.type === "cluster")
        if(use_inside_line){
        clusters
            .append("path")
            .attr("class", "action-start-line-1")
            .attr("d", d => d.start_line)
            // .style("stroke-dasharray", "4 4 8 4 12 4")
            .style("fill", "none")
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .attr("transform", d => `translate(0, ${d.width * beta})`)

        clusters
            .append("path")
            .attr("class", "action-center-line-1")
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .style("fill", "none")
            .attr("d", d => d.center_line_1)
            // .attr("transform", d => `translate(0, ${d.width * beta})`)

        clusters
            .append("path")
            .attr("class", "action-end-line-1")
            .style("stroke-width", d => d.width * omg)
            // .style("stroke-dasharray", "4 4 8 4 12 4")
            .style("stroke", Global.MidBlue)
            .style("fill", "none")
            .attr("d", d => d.end_line)
            .attr("transform", d => `translate(0, ${d.width * beta})`)
        
        clusters
            .append("path")
            .attr("class", "action-start-line-2")
            .attr("d", d => d.start_line)
            // .style("stroke-dasharray", "4 4 8 4 12 4")
            .style("fill", "none")
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .attr("transform", d => `translate(0, ${- d.width * beta})`)

        clusters
            .append("path")
            .attr("class", "action-center-line-2")
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .style("fill", "none")
            .attr("d", d => d.center_line_2)
            // .attr("transform", d => `translate(0, ${- d.width * beta})`)

        create
            .append("path")
            .attr("class", "action-end-line-2")
            .style("stroke-width", d => d.width * omg)
            // .style("stroke-dasharray", "4 4 8 4 12 4")
            .style("stroke", Global.MidBlue)
            .style("fill", "none")
            .attr("d", d => d.end_line)
            .attr("transform", d => `translate(0, ${- d.width * beta})`)
        }
        
        if(!that.cards[0].normal_layout){
            let label = create.append("g")
                .attr("class", "label-group")
                .style("opacity", d => d.selected ? 1 : 0)

            label
                .selectAll(".label")
                .data(d => d.frames.filter(d => d.is_action > -1), d => d.id)
                .enter().append("text")
                .attr("class", "label")
                .attr("transform", d => `translate(${d.x}, ${d.y})`)
                .attr("dx", -30)
                .attr("dy", -5)
                .attr("text-anchor", "middle")
                .attr("font-weight", 650)
                .style("opactity", 0)
                .text(d => `${d.video_left_bound + d.frame_idx}`)
        }

        create
            .on("mouseover", function (ev, d) {
                if(that.parent.zoom_in || that.parent.change_rep) that.cluster_highlight(d.parent.clusters[d.cluster_idx].cluster_id);
                d3.select(this).selectAll("path")
                    .transition()
                    .style("stroke", Global.Orange)
                    .duration(Global.QuickAnimation)
            })
            .on("mouseout", function (ev, d) {
                d3.select(this).selectAll("path")
                    .transition()
                    .style("stroke", d => that.stroke_color(d))
                    .duration(Global.QuickAnimation)
                d3.select(this).selectAll("path.action-start-line-1")
                    .transition()
                    .style("stroke", d => Global.MidBlue)
                    .duration(Global.QuickAnimation)
                d3.select(this).selectAll("path.action-center-line-1")
                    .transition()
                    .style("stroke", d => Global.MidBlue)
                    .duration(Global.QuickAnimation)
                d3.select(this).selectAll("path.action-end-line-1")
                    .transition()
                    .style("stroke", d => Global.MidBlue)
                    .duration(Global.QuickAnimation)
                d3.select(this).selectAll("path.action-start-line-2")
                    .transition()
                    .style("stroke", d => Global.MidBlue)
                    .duration(Global.QuickAnimation)
                d3.select(this).selectAll("path.action-center-line-2")
                    .transition()
                    .style("stroke", d => Global.MidBlue)
                    .duration(Global.QuickAnimation)
                d3.select(this).selectAll("path.action-end-line-2")
                    .transition()
                    .style("stroke", d => Global.MidBlue)
                    .duration(Global.QuickAnimation)
            })
            // .on("dblclick", function(ev, d){
            //     that.parent.set_selected_cluster({
            //         selected_action: d.rep_action.id || d.rep_action
            //     })
            //     // that.sub_component_view_update();
            // })
            .on("click", function (ev, d) {
                if (that.parent.compare_to) {
                    that.last_action = d
                    that.parent.set_selected_cluster({
                        selected_action: d.rep_action.id || d.rep_action,
                    })
                    that.parent.compare_to = false;
                    return
                }
                if (that.cards[0].normal_layout) {
                    that.cards[0].rep_frames.forEach(e => {
                        if (e.cluster_idx == d.cluster_idx) {
                            let flag = 0
                            e.frames.forEach((f, index) => {
                                if (f.is_action > 0 && f.action_idx == d.action_idx) {
                                    e.select_image_index = index
                                    f.is_selected = 1
                                    flag = 1
                                }
                            })
                            if (flag) {
                                e.frames.forEach((f) => {
                                    if (f.action_idx != d.action_idx) {
                                        f.is_selected = 0
                                    }
                                })
                            }
                        }
                    })
                    that.frame_band_subcomponent_update(that.aligns_group, that.cards[0].rep_frames);
                    that.frame_point_subcomponent_update(that.aligns_group, that.cards[0].clusters);
                } else {
                    if (that.cards[0].current_action) {
                        that.cards[0].current_action.selected = 0
                        that.cards[0].current_action = null
                    }
                    d.selected = !d.selected
                    if (d.selected) {
                        that.cards[0].current_action = d
                    }
                    that.sub_component_view_refresh()
                }
            })

        return create;
    };

    that.single_action_line_update = function(action_id, ani = false) {
        let action_elem = that.aligns_group
            .selectAll(".action-group")
            .data(that.cards[0].actions, d => d.action_id)
            .filter(d => d.action_id == action_id)

        let cluster_elem = that.aligns_group
            .selectAll(".cluster-frame-group")
            .data(that.cards[0].clusters, d => d.cluster_id)

        let frame_elem = cluster_elem
            .selectAll(".frame-group")
            .data(d => d.frames, d => d.id)
            .filter(d => d.action_id == action_id)

        action_elem.select(".label-group")
            .style("opacity", d => d.selected ? 1 : 0)

        // TODO: labels when update
        let labels = action_elem.select(".label-group")
            .selectAll(".label")
            .data(d => d.frames.filter(d => d.is_action > -1), d => d.id)

        labels
            .enter().append("text")
            .attr("class", "label")
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
            .attr("dx", -30)
            .attr("dy", -5)
            .attr("text-anchor", "middle")
            .attr("font-weight", 650)
            .text(d => `${d.video_left_bound + d.frame_idx}`)
            .style("opacity", 0)
            .transition()
            .duration(that.create_ani)
            .style("opacity", 1)
        
        labels
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
            .attr("dx", -30)
            .attr("dy", -5)
            .attr("text-anchor", "middle")
            .attr("font-weight", 650)
            .text(d => `${d.video_left_bound + d.frame_idx}`)
            .style("opacity", d => d.is_hidden ? 0 : 1)

        labels
            .exit()
            .transition()
            .duration(that.remove_ani)
            .style("opacity", 0)
            .remove();

        if (ani) {
            console.log('update line with animation')
            frame_elem.style("display", d => d.is_hidden ? "none" : "block")

            action_elem
                .select("path.action-start-line")
                .style("stroke-width", d => d.width)
                .style("stroke", d => that.stroke_color(d))
                .style("opacity", 1)
                .transition()
                .duration(Global.QuickAnimation)
                .attrTween("d", function(d){
                    return pathInterpolate(d3.select(this).attr("d"), d.start_line)
                })
    
            action_elem
                .select("path.action-center-line")
                .style("stroke-width", d => d.width)
                .style("stroke", d => that.stroke_color(d))
                .style("opacity", 1)
                .transition()
                .duration(Global.QuickAnimation)
                .attrTween("d", function(d){
                    return pathInterpolate(d3.select(this).attr("d"), d.center_line)
                })
                //.attr("d", d => d.center_line)
    
            action_elem
                .select("path.action-end-line")
                .style("stroke-width", d => d.width)
                .style("stroke", d => that.stroke_color(d))
                .style("opacity", 1)
                .transition()
                .duration(Global.QuickAnimation)
                .attrTween("d", function(d){
                    return pathInterpolate(d3.select(this).attr("d"), d.end_line)
                })
            
            // new line
            action_elem
                .select("path.action-start-line-1")
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .style("opacity", 1)
                .attr("transform", d => `translate(0, ${d.width * beta})`)
                .transition()
                .duration(Global.QuickAnimation)
                .attrTween("d", function(d){
                    return pathInterpolate(d3.select(this).attr("d"), d.start_line)
                })
    
            action_elem
                .select("path.action-center-line-1")
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .style("opacity", 1)
                // .attr("transform", d => `translate(0, ${d.width * beta})`)
                .transition()
                .duration(Global.QuickAnimation)
                .attrTween("d", function(d){
                    return pathInterpolate(d3.select(this).attr("d"), d.center_line_1)
                })
                //.attr("d", d => d.center_line)
    
            action_elem
                .select("path.action-end-line-1")
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .style("opacity", 1)
                .attr("transform", d => `translate(0, ${d.width * beta})`)
                .transition()
                .duration(Global.QuickAnimation)
                .attrTween("d", function(d){
                    return pathInterpolate(d3.select(this).attr("d"), d.end_line)
                })

            action_elem
                .select("path.action-start-line-2")
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .style("opacity", 1)
                .attr("transform", d => `translate(0, ${-d.width * beta})`)
                .transition()
                .duration(Global.QuickAnimation)
                .attrTween("d", function(d){
                    return pathInterpolate(d3.select(this).attr("d"), d.start_line)
                })
    
            action_elem
                .select("path.action-center-line-2")
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .style("opacity", 1)
                // .attr("transform", d => `translate(0, ${-d.width * beta})`)
                .transition()
                .duration(Global.QuickAnimation)
                .attrTween("d", function(d){
                    return pathInterpolate(d3.select(this).attr("d"), d.center_line_2)
                })
                //.attr("d", d => d.center_line)
    
            action_elem
                .select("path.action-end-line-2")
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .style("opacity", 1)
                .attr("transform", d => `translate(0, ${-d.width * beta})`)
                .transition()
                .duration(Global.QuickAnimation)
                .attrTween("d", function(d){
                    return pathInterpolate(d3.select(this).attr("d"), d.end_line)
                })

            

        } else {
            // console.log("frame_elem", frame_elem)
            frame_elem.style("display", d => d.is_hidden ? "none" : "block")
            
            action_elem
                .select("path.action-start-line")
                .attr("d", d => d.start_line)
                .style("stroke-width", d => d.width)
                .style("stroke", d => that.stroke_color(d))
                .style("opacity", 1)
    
            action_elem
                .select("path.action-center-line")
                .attr("d", d => d.center_line)
                .style("stroke-width", d => d.width)
                .style("stroke", d => that.stroke_color(d))
                .style("opacity", 1)
    
            action_elem
                .select("path.action-end-line")
                .attr("d", d => d.end_line)
                .style("stroke-width", d => d.width)
                .style("stroke", d => that.stroke_color(d))
                .style("opacity", 1)

            //new line
            action_elem
                .select("path.action-start-line-1")
                .attr("d", d => d.start_line)
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .attr("transform", d => `translate(0, ${d.width * beta})`)
                .style("opacity", 1)
    
            action_elem
                .select("path.action-center-line-1")
                .attr("d", d => d.center_line_1)
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                // .attr("transform", d => `translate(0, ${d.width * beta})`)
                .style("opacity", 1)
    
            action_elem
                .select("path.action-end-line-1")
                .attr("d", d => d.end_line)
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .attr("transform", d => `translate(0, ${d.width * beta})`)
                .style("opacity", 1)

            action_elem
                .select("path.action-start-line-2")
                .attr("d", d => d.start_line)
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .attr("transform", d => `translate(0, ${-d.width * beta})`)
                .style("opacity", 1)
    
            action_elem
                .select("path.action-center-line-2")
                .attr("d", d => d.center_line_2)
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                // .attr("transform", d => `translate(0, ${-d.width * beta})`)
                .style("opacity", 1)
    
            action_elem
                .select("path.action-end-line-2")
                .attr("d", d => d.end_line)
                .style("stroke-width", d => d.width * omg)
                .style("stroke", Global.MidBlue)
                .attr("transform", d => `translate(0, ${-d.width * beta})`)
                .style("opacity", 1)
        }
    }

    that.action_line_update = function (elem) {
        let is_zoom_in = false
        if (that.last_card && that.cards[0].depth - 1 == that.last_card.depth) {
            is_zoom_in = true
        }

        let update_delay
        // console.log('is_zoom_in', is_zoom_in)
        if (is_zoom_in) {
            update_delay = that.remove_ani
        } else {
            update_delay = that.remove_ani * 2 + Global.QuickAnimation
        }

        console.log('action_line_update', elem)
        elem.transition()
            .duration(that.update_ani)
            // .delay(update_delay)
            .attr("transform", d => {
                // console.log('transform', d)
                return `translate(${d.x}, ${d.y})`
            })
        
        elem.transition()
            .duration(that.update_ani)
            .delay(that.remove_ani + that.update_ani)
            .style("opacity", d => d.type == 'fake' ? 0 : 1)

        let label_group = elem.select(".label-group")

        label_group
            .transition()
            .duration(that.update_ani)
            // .delay(that.remove_ani)
            .style("opacity", d => d.selected ? 1 : 0)

        label_group
            .selectAll(".label")
            .data(d => {
                let data = d.frames.filter(d => d.is_action > -1);
                data.forEach(e => e.is_anchor=d.is_anchor);
                console.log("debug", data, d);
                return data;
            }, d => d.id)
            .transition()
            .duration(that.update_ani)
            .delay(update_delay)
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
            
        let labels = label_group
            .selectAll(".label")
            .data(d => d.frames.filter(d => d.is_action > -1), d => d.id)

        labels
            .enter().append("text")
            .attr("class", "label")
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
            .attr("dx", -30)
            .attr("dy", d => d.is_anchor ? -5 : 0)
            .attr("text-anchor", "middle")
            .attr("font-weight", 650)
            .text(d => `${d.video_left_bound + d.frame_idx}`)
        
        labels
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
            .attr("dx", -30)
            .attr("dy", d => d.is_anchor ? -5 : 5)
            .attr("text-anchor", "middle")
            .attr("font-weight", 650)
            .text(d => `${d.video_left_bound + d.frame_idx}`)

        const opa = 0.6
        elem.select("path.action-start-line")
            .transition()
            .duration(that.update_ani)
            .delay(update_delay)
            .attrTween("d", function(d){
                return pathInterpolate(d3.select(this).attr("d"), d.start_line)
            })
            .style("stroke-width", d => d.width)
            .style("stroke", d => that.stroke_color(d))
            .style("opacity", d => d.type === "cluster" ? opa: 1)

        elem.select("path.action-center-line")
            .transition()
            .duration(that.update_ani)
            .delay(update_delay)
            .style("stroke-width", d => d.width)
            .style("stroke", d => that.stroke_color(d))
            .style("opacity", d => d.type === "cluster" ? opa: 1)
            .attrTween("d", function(d){
                return pathInterpolate(d3.select(this).attr("d"), d.center_line)
            })
            .attr("d2", d => d.center_line_z)

        elem.select("path.action-end-line")
            .transition()
            .duration(that.update_ani)
            .delay(update_delay)
            .attrTween("d", function(d){
                return pathInterpolate(d3.select(this).attr("d"), d.end_line)
            })
            .style("stroke-width", d => d.width)
            .style("stroke", d => that.stroke_color(d))
            .style("opacity", d => d.type === "cluster" ? opa: 1)

        // new line
        elem.select("path.action-start-line-1")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attrTween("d", function(d){
                return pathInterpolate(d3.select(this).attr("d"), d.start_line)
            })
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .style("opacity", 1)
            .attr("transform", d => `translate(0, ${d.width * beta})`)
        elem.select("path.action-center-line-1")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .style("opacity", 1)
            .attrTween("d", function(d){
                return pathInterpolate(d3.select(this).attr("d"), d.center_line_1)
            })
            // .attr("transform", d => `translate(0, ${d.width * beta})`)
        elem.select("path.action-end-line-1")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attrTween("d", function(d){
                return pathInterpolate(d3.select(this).attr("d"), d.end_line)
            })
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .style("opacity", 1)
            .attr("transform", d => `translate(0, ${d.width * beta})`)
        elem.select("path.action-start-line-2")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attrTween("d", function(d){
                return pathInterpolate(d3.select(this).attr("d"), d.start_line)
            })
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .style("opacity", 1)
            .attr("transform", d => `translate(0, ${-d.width * beta})`)
        elem.select("path.action-center-line-2")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .style("opacity", 1)
            .attrTween("d", function(d){
                return pathInterpolate(d3.select(this).attr("d"), d.center_line_2)
            })
            // .attr("transform", d => `translate(0, ${-d.width * beta})`)
        elem.select("path.action-end-line-2")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attrTween("d", function(d){
                return pathInterpolate(d3.select(this).attr("d"), d.end_line)
            })
            .style("stroke-width", d => d.width * omg)
            .style("stroke", Global.MidBlue)
            .style("opacity", 1)
            .attr("transform", d => `translate(0, ${-d.width * beta})`)

        elem.select("rect.action-rect")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("width", d => d.width)
            .attr("height", d => d.height);
    };

    that.action_line_remove = function (elem) {
        elem
            .exit()
            .transition()
            .duration(that.remove_ani)
            .style("opacity", 0)
            .remove();
    };

    that.frame_band_subcomponent_update = function (groups, data) {

        console.log('frame_band_subcomponent_update', data)
        let elem;
        //if (!that.data.is_anchor_alignment){
            elem = groups
                .selectAll(".key-frame-group")
                .data(data, d => d.band_id);
        /*}
        else{
            elem = groups
                .selectAll(".key-frame-group")
                .data(data, d => d.frames[0].id);
        }*/

        let create = elem.enter()
            .append("g")
            .attr("class", "key-frame-group")
            .attr("id", d => "key-frame-" + d.band_id)
            .attr("transform", d => `translate(${d.offset.x}, ${d.offset.y})`)

        create
            .style("opacity", 0)
            .transition()
            .duration(that.create_ani)
            .delay(that.update_ani + that.remove_ani + that.create_ani)
            .style("opacity", 1);

        create
            .append("rect")
            .attr("class", "key-frame-band")
            .attr("x", d => d.x - d.width / 2 + 2.5)
            .attr("y", d => d.y)
            .attr("width", d => d.width - 5)
            .attr("height", d => d.height - 5)
            .style("fill", d3.interpolate('white', Global.Blue)(.4))
            .style("opacity", 1)
            .style("stroke", "white")
            .style("stroke-width", 2.5)
            .style("display", d => d.show_band ? "block" : "none")

        // elem = elem.merge(create)
        // elem.select('.extend-image').remove()

        let extend_image = create
            .append('g')
            .attr('class', 'extend-image')
            .attr('transform', d => `translate(${d.x - d.image_offset.x - that.frame_width / 2}, ${d.y - d.image_offset.y - that.frame_height})`)
            .style('opacity', 0)
            .style('display', 'none')

        extend_image.append("line")
            .attr("class", "key-frame-line")
            .style("stroke-width", 2)
            .style("stroke", "#bfbfbf")
            .style("opacity", 1)
            .attr("x1", 0)
            .attr("x2", d => -d.image_offset.x)
            .attr("y1", 0)
            .attr("y2", d => -d.image_offset.y)
            .attr('transform', d => `translate(${d.image_offset.x + that.frame_width / 2},${d.image_offset.y + that.frame_height})`)

        extend_image.append('image')
            .attr('width', that.frame_width)
            .attr('height', that.frame_height)
            .attr('xlink:href', d => d.select_image_index >= 0 ?
                `${that.parent.server_url}/Frame/single_frame?filename=${d.frames[d.select_image_index].id}` :
                '')

        setTimeout(() => {
            elem.select("g.extend-image")
                .select("image")
                .attr('xlink:href', d => d.select_image_index >= 0 ?
                    `${that.parent.server_url}/Frame/single_frame?filename=${d.frames[d.select_image_index].id}` :
                    '')
        }, that.remove_ani + that.update_ani + that.create_ani + Global.QuickAnimation)

        extend_image.append('rect')
            .attr('width', that.frame_width)
            .attr('height', that.frame_height)
            .style("fill", "none")
            .style("stroke-width", 4)
            .style("stroke", "#bfbfbf")
            .style("opacity", 1)

        let extend_create = create
            .append("g")
            .attr("class", "extend-lines")

        extend_create
            .selectAll(".extend-line")
            .data(d => d.frames)
            .enter()
            .append("line")
            .attr("class", "extend-line")
            .attr("x1", d => d.x1)
            .attr("y1", d => d.y1)
            .attr("x2", d => d.x2)
            .attr("y2", d => d.y2)
            .style("stroke-width", 4)
            .style("stroke", "#bfbfbf")
            .style("opacity", 1)

        elem.transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("transform", d => `translate(${d.offset.x}, ${d.offset.y})`)
        
        elem.select("rect.key-frame-band")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("x", d => d.x - d.width / 2 + 2.5)
            .attr("y", d => d.y)
            .attr("width", d => d.width - 5)
            .attr("height", d => d.height - 5)
            .style("fill",  d3.interpolate('white', Global.Blue)(.4))
            .style("display", d => d.show_band ? "block" : "none")

        
        if(!that.data.is_anchor_alignment){
            elem.merge(create)
                .select("g.extend-image")
                .transition()
                .duration(that.update_ani)
                .delay(that.remove_ani)
                .attr('transform', d => `translate(${d.x - d.image_offset.x - that.frame_width / 2}, ${d.y - d.image_offset.y - that.frame_height})`)
                .style("opacity", d => d.show_band ? 1 : 0)
                .style('display', function(d){
                    return !d.show_band ? 'none' : 'block' })
        } else{
            elem.merge(create)
                .select("g.extend-image")
                .transition()
                .duration(that.update_ani)
                .delay(that.remove_ani)
                .attr('transform', d => `translate(${d.x - d.image_offset.x - that.frame_width / 2}, ${d.y - d.image_offset.y - that.frame_height})`)
                .style("opacity", d => d.show_image ? 1 : 0)
                .style('display', function(d){
                    return !d.show_image ? 'none' : 'block' })
        }
    
        elem.select("line.key-frame-line")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("x2", d => -d.image_offset.x)
            .attr("y2", d => -d.image_offset.y)
            .attr('transform', d => `translate(${d.image_offset.x + that.frame_width / 2},${d.image_offset.y + that.frame_height})`)
                
        elem.select("line.extend-line")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("x1", d => d.x1)
            .attr("y1", d => d.y1)
            .attr("x2", d => d.x2)
            .attr("y2", d => d.y2)

        elem.exit()
            .transition()
            .duration(that.remove_ani)
            .style("opacity", 0)
            .remove();
    };

    that.upper_background_update = function(groups, data) {
        groups.selectAll('.card-upper-background').remove()
        let action_offset = data.actions[0]

        let background = groups.append("g")
            .attr("class", "card-upper-background")
            .attr("transform", `translate(${action_offset.x},${action_offset.y})`)

        background.append('rect')
            .attr('class', 'align')
            .attr('x', 0)
            .attr('y', 0)
            .attr('rx', 5)
            .attr('ry', 5)
            .attr('width', 1)
            .attr('height', 1)
            .attr('stroke-dasharray', '5 5')
            .attr('stroke-width', '3.5px')
            .attr('stroke', Global.DarkBlueGray)
            .attr('fill', 'none')
            .style('opacity', 0)
    }

    that.lower_background_update = function(groups, data) {
        groups.selectAll('.card-lower-background').remove()
        let action_offset = data.actions[0]

        let background = groups.append("g")
            .attr("class", "card-lower-background")
            .attr("transform", `translate(${action_offset.x},${action_offset.y})`)

        background.lower()
            //.style('opacity', 1)

        if (!data.normal_layout) {
            let must_link_area = background
                .append('g')
                .attr('transform', `translate(${data.must_link_area.x1}, ${data.must_link_area.y1})`)
                .attr('class', 'must-link')
                .style('opacity', 0)

            must_link_area
                .append('text')
                .attr('text-anchor', 'end')
                .attr('dx', -10)
                .attr('dy', (data.must_link_area.y2 - data.must_link_area.y1) / 2)
                .text('Must-links')

            must_link_area
                .append('rect')
                .attr('rx', 10)
                .attr('ry', 10)
                .attr('width', data.must_link_area.x2 - data.must_link_area.x1)
                .attr('height', data.must_link_area.y2 - data.must_link_area.y1)
                .attr('stroke', "none")
                .attr('fill', Global.LightBlue)

            let must_not_link_area = background
                .append('g')
                .attr('transform', `translate(${data.must_not_link_area.x1}, ${data.must_not_link_area.y1})`)
                .attr('class', 'must-not-link')
                .style('opacity', 0)

            must_not_link_area
                .append('text')
                .attr('text-anchor', 'end')
                .attr('dx', -10)
                .attr('dy', (data.must_not_link_area.y2 - data.must_not_link_area.y1) / 2)
                .text('Cannot-links')

            must_not_link_area
                .append('rect')
                .attr('rx', 10)
                .attr('ry', 10)
                .attr('width', data.must_not_link_area.x2 - data.must_not_link_area.x1)
                .attr('height', data.must_not_link_area.y2 - data.must_not_link_area.y1)
                .attr('stroke', "none")
                .attr('fill', Global.LightBlue)
        }
    }

    that.cluster_subcomponent_update = function (groups, data) {
        // console.log('cluster_id', data.map(d => d.cluster_id))
        let elem = groups
            .selectAll(".cluster-group")
            .data(data, d => d.cluster_id);

        let action_clusters = that.single_cluster_create(elem);
        let merged_cluster_groups = elem.merge(action_clusters);
        that.single_cluster_update(elem);
        that.single_cluster_remove(elem);
        that.barchart_subsubcomponent_update(
            merged_cluster_groups.selectAll(".cluster-bar-chart"));
    };

    that.single_cluster_create = function (elem) {
        let create = elem.enter()
            .append("g")
            .attr("class", "cluster-group")
            .attr("id", d => "cl-" + d.cluster_id)
            .attr("transform", d => `translate(${d.x}, ${d.y})`);

        create.each(d => {
            let old_item = d3.select(`#ag-N${d.cluster_id}`)
            if (!old_item.empty() && !old_item.select('.action-center-line').empty()) {
                d.old_path = old_item.select('.action-center-line').attr('d2')
                const points = d.old_path.split(/[a-zA-Z ]/g).slice(1)
                const splits = d.old_path.match(/[a-zA-Z ]/g)
                const valid_points = points.filter(d => d.length > 0)
                // console.log('valid_points', valid_points)
                if(d.points.length == 0 || valid_points.length == 0){
                    d.old_path = null;
                    return;
                }
                const old_x = valid_points.map(d => parseFloat(d.split(',')[0])).reduce((a, b) => a + b) / valid_points.length
                const old_y = valid_points.map(d => parseFloat(d.split(',')[1])).reduce((a, b) => a + b) / valid_points.length
                const new_x = d.points.map(d => d[0]).reduce((a, b) => a + b) / d.points.length
                const new_y = d.points.map(d => d[1]).reduce((a, b) => a + b) / d.points.length
                const dx = new_x - old_x
                const dy = new_y - old_y
                d.mid_path = ''
                for (let i = 0; i < points.length; ++i) {
                    d.mid_path += splits[i]
                    if (points[i].length == 0) {
                        d.mid_path += points[i]
                    } else {
                        const x = parseFloat(points[i].split(',')[0])
                        const y = parseFloat(points[i].split(',')[1])
                        d.mid_path += `${x+dx},${y+dy}`
                    }
                }

            }
        })

        create
            .filter(d => d.old_path)
            .style("opacity", 0)
            .transition()
            .duration(that.remove_ani)
            .style("opacity", 1)
        
        create
            .filter(d => !d.old_path)
            .style("opacity", 0)
            .transition()
            .duration(that.create_ani)
            .delay(that.update_ani + that.remove_ani + that.create_ani)
            .style("opacity", 1);
        
        create
            .append("path")
            .attr("class", "cluster-hull")
            .attr("id", d => "cl-h-" + d.cluster_id)
            .style("stroke", "none")
            .style("fill", d => d.old_path ? Global.Blue60 : "rgba(1,103,243,0.1)")
            .attr("d", d => d.old_path ? d.old_path : d.convex_hull)

        create
            .select(".cluster-hull")
            .filter(d => d.old_path)
            .transition()
            .duration(that.create_ani)
            .delay(that.remove_ani)
            .style("fill", "rgba(1,103,243,0.2)")
            .attrTween("d", function (d) {
                return pathInterpolate(d.old_path, d.mid_path)
            })
            .transition()
            .duration(that.update_ani)
            // .delay(that.remove_ani + that.create_ani)
            .style("fill", "rgba(1,103,243,0.1)")
            .attrTween("d", function (d) {
                return pathInterpolate(d.mid_path, d.convex_hull)
            })

        create.lower()

        create
            .on("click", (ev, d) => {
                if (that.parent.change_rep){
                    let card = that.data;
                    if(card.cluster_rep_frame_indexes.indexOf(d.cluster_idx) === -1){
                        let mdelta = Math.abs(card.cluster_rep_frame_indexes[0] - d.cluster_idx);
                        let mindex = 0;
                        let mvalue = card.cluster_rep_frame_indexes[0];
                        
                        card.cluster_rep_frame_indexes.forEach((value, index) => {
                            let tdelta = Math.abs(value - d.cluster_idx)
                            if(tdelta > mdelta || (tdelta == mdelta && value > mvalue)){
                                mdelta = tdelta;
                                mindex = index;
                                mvalue = value;
                            }
                        })
                        card.cluster_rep_frame_indexes[mindex] = d.cluster_idx;
                        that.sub_component_view_update();
                    }
                    that.parent.change_rep = false;
                    that.cluster_dehighlight(d.cluster_id);
                    return;
                }
                if (!that.parent.zoom_in || !that.data.all_children || !that.data.all_children[d.cluster_idx].all_children) return;
                that.data = that.data.all_children[d.cluster_idx];
                that.parent.push_history({id: d.cluster_idx, is_comparison: false})
                that.stack.push(that.data);
                that.sub_component_view_update();
                that.parent.zoom_in = false;
            })
            .on("mouseover", function(ev, d) {
                if(that.parent.zoom_in || that.parent.change_rep) that.cluster_highlight(d.cluster_id);
            })
            .on("mouseout", function(ev, d) {
                if(that.parent.zoom_in || that.parent.change_rep) that.cluster_dehighlight(d.cluster_id);
            });

        let barchart_group = create
            .append("g")
            .attr("class", "cluster-bar-chart")
            .attr("transform", d => `translate(${d.bar_chart_x + 100 || 0}, ${d.bar_chart_y - d.bar_height + d.cur_height / 2 - 10 || 0})`)
            .style("display", d => d.bars.length ? "block" : "none")
            .style("opacity", 0)

        barchart_group
            .transition()
            .duration(Global.QuickAnimation)
            .delay(that.update_ani + that.create_ani + that.remove_ani)
            .style("opacity", 1)

        let texts_create = that.aligns_group
            .selectAll("g.legend-text")
            .data([null])
            .enter()
            .append("g")
            .attr("class", "legend-text")
            .attr("transform", `translate(${that.layout_width - 218}, -30)`)
            .style("opacity", 0);
        texts_create.append("text")
            .attr("class", "key-text")
            .text("Action length (0.64 s/frame)")
            .attr("font-weight", that.in_length_compare ? "bold": "normal")
            .attr("fill", Global.DeepGray);

        let legend_create = texts_create.append("g")
            .attr("class", "legend-inside")
            .style("opacity", that.in_length_compare ? 1: 0);
        legend_create.append("rect")
            .attr("x", 35)
            .attr("y", 14)
            .attr("width", 40)
            .attr("height", 14)
            .attr("fill", Global.Blue40);
        legend_create.append("text")
            .text("Current")
            .attr("transform", `translate(110, 27)`)
            .attr("fill", Global.Blue40)
            .attr("stroke", Global.Blue40);
        legend_create.append("rect")
            .attr("x", 35)
            .attr("y", 39)
            .attr("width", 40)
            .attr("height", 14)
            .attr("fill", Global.Blue40)
            .style("opacity", 0.5);
        legend_create.append("text")
            .text("Previous")
            .attr("transform", `translate(110, 52)`)
            .attr("fill", Global.Blue40)
            .attr("stroke", Global.Blue40)
            .style("opacity", 0.5);

        let texts = that.aligns_group
            .select("g.legend-text");
        let legend = texts.select("g.legend-inside");
        texts.transition()
            .duration(that.create_ani)
            .delay(that.remove_ani + that.create_ani)
            .style("opacity", that.parent.history.length > 0 && 
                        that.parent.history[ that.parent.history.length - 1].is_comparison ? 0: 1)
        // texts.select("text.key-text")
        //     .transition()
        //     .duration(that.create_ani)
        //     .attr("font-weight", that.in_length_compare ? "bold": "normal")
        legend.transition()
            .duration(that.create_ani)
            .style("opacity", that.in_length_compare ? 1: 0);

        barchart_group
            .append("rect")
            .attr("class", "barchart-underline")
            .attr("x", -5)
            .attr("y", d => d.bar_height)
            .attr("width", d => d.bar_width + 10 || 0)
            .attr("height", 2)
            .style("fill", Global.Blue40);

        return create;
    };

    that.single_cluster_update = function (elem) {
        elem.lower()

        elem.transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
        
        elem.select("path.cluster-hull")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("d", d => d.convex_hull)
        
        elem.select("g.cluster-bar-chart")
            .style("display", d => d.bars.length ? "block" : "none")
            .style("opacity", 0)
            .transition()
            .duration(that.update_ani)
            .delay(that.update_ani + that.create_ani + that.remove_ani)
            .attr("transform", d => `translate(${d.bar_chart_x + 100 || 0}, ${d.bar_chart_y - d.bar_height + d.cur_height / 2 - 10 || 0})`)
            .style("opacity", 1)

        elem.select("rect.barchart-underline")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("x", -5)
            .attr("y", d => d.bar_height)
            .attr("width", d => d.bar_width + 10 || 0);
    };

    that.single_cluster_remove = function (elem) {
        const elem1 = elem.exit().filter(d => !d.old_path || !d.mid_path || !that.cards[0].actions.find(e => e.action_id.slice(1) == d.cluster_id))
        const elem2 = elem.exit().filter(d => d.old_path && d.mid_path && that.cards[0].actions.find(e => e.action_id.slice(1) == d.cluster_id))

        elem1
            .transition()
            .duration(that.remove_ani)
            .delay(that.remove_ani)
            .style("opacity", 0)
            .remove();

        elem2
            .select("g.cluster-bar-chart")
            .transition()
            .duration(that.remove_ani)
            .style("opacity", 0)
            .remove();
        
        elem2.select(".cluster-hull")
            .transition()
            .duration(Global.QuickAnimation)
            .delay(that.remove_ani * 2)
            .style("fill", "rgba(1,103,243,0.1)")
            .attrTween("d", function (d) {
                return pathInterpolate(d.convex_hull, d.mid_path)
            })
            .transition()
            .duration(that.update_ani)
            .style("fill", Global.Blue40)
            .attrTween("d", function (d) {
                return pathInterpolate(d.mid_path, d.old_path)
            })
            .transition()
            .duration(that.remove_ani)
            .delay(that.create_ani * 2 - Global.QuickAnimation - that.remove_ani * 2)
            .style('opacity', 0)

        elem2
            .transition()
            .duration(that.remove_ani)
            .delay(that.update_ani + that.create_ani * 2)
            .style("opacity", 0)
            .remove();
        
    };

    that.barchart_subsubcomponent_update = function (elem) {
        elem.style("display", d => d.bars.length > 0 ? "block" : "none")
        // console.log("elem", elem);
        let bar_groups = elem.selectAll(".action-bar")
            .data(d => d.bars);
        // console.log("bar_groups", bar_groups);
        // create
        let bars = bar_groups.enter()
            .append('g')
            .attr("class", "action-bar");
        bars.append('rect')
            .attr("class", "action-bar-rect")
            .attr("x", d => d.x)
            .attr("y", d => d.y)
            .attr("width", d => d.width)
            .attr("height", d => d.height)
            .attr("fill", Global.Blue40);
        bars.append('rect')
            .attr("class", "action-compare-bar-rect")
            .attr("x", d => that.in_length_compare ? d.cx: d.x)
            .attr("y", d => that.in_length_compare ? d.cy: d.y)
            .attr("width", d => d.width)
            .attr("height", d => that.in_length_compare ? d.cheight: d.height)
            .attr("fill", Global.Blue40)
            .style("opacity", that.in_length_compare ? 0.5: 0);
        bars.append('text')
            .attr("class", "bar-label")
            .attr("x", d => d.x + 2)
            .attr("y", d => d.y + d.height + 18)
            .text(d => `${d.left}`);
        console.log("bar_groups", bar_groups);

        // update
        bar_groups.select("rect.action-bar-rect")
            .transition()
            .duration(that.update_ani)
            .attr("x", d => d.x)
            .attr("y", d => d.y)
            .attr("width", d => d.width)
            .attr("height", d => d.height);

        bar_groups.select("rect.action-compare-bar-rect")
            .transition()
            .duration(that.update_ani)
            .attr("x", d => that.in_length_compare ? d.cx: d.x)
            .attr("y", d => that.in_length_compare ? d.cy: d.y)
            .attr("width", d => d.width)
            .attr("height", d => that.in_length_compare ? d.cheight: d.height)
            .style("opacity", that.in_length_compare ? 0.5: 0);

        bar_groups.select("text.bar-label")
            .transition()
            .duration(that.update_ani)
            .attr("x", d => d.x + 2)
            .attr("y", d => d.y + d.height + 18)
            .text(d => `${d.left}`);

        // remove
        bar_groups.exit()
            .remove();
    }

    // that.cluster_frame_subcomponent_update = function(groups, data){
    that.frame_point_subcomponent_update = function (groups, data) {
        let is_zoom_in = false
        if (that.last_card && that.cards[0].depth - 1 == that.last_card.depth) {
            is_zoom_in = true
        }

        let update_delay
        // console.log('is_zoom_in', is_zoom_in)
        if (is_zoom_in) {
            update_delay = that.remove_ani
        } else {
            update_delay = that.remove_ani * 2 + Global.QuickAnimation
        }
        
        let elem = groups
            .selectAll(".cluster-frame-group")
            .data(data, d => d.cluster_id)

        let create = elem.enter()
            .append("g")
            .attr("class", "cluster-frame-group")
            .attr("id", d => "cl-" + d.cluster_id)
            .attr("transform", d => `translate(${d.x}, ${d.y})`);

        create
            .style("opacity", 0)
            .transition()
            .duration(that.create_ani)
            .delay(that.update_ani + that.remove_ani + that.create_ani)
            .style("opacity", 1);

        elem.transition()
            .duration(that.update_ani)
            .delay(update_delay)
            .attr("transform", d => `translate(${d.x}, ${d.y})`)

        elem
            .exit()
            .transition()
            .duration(that.remove_ani)
            .style("opacity", 0)
            .remove()

        let merged_action_groups = elem.merge(create)
        // TODO: change it to single_frame_subcomponent_update
        that.frame_subcomponent_update(merged_action_groups);
    };


    that.frame_subcomponent_update = function (groups) {
        let elem = groups
            .selectAll(".frame-group")
            .data(d => d.frames, d => d.id)
        //.filter(d => d.is_action >= 0)
        that.frame_create(elem);
        that.frame_update(elem);
        that.frame_remove(elem);
    };
// that.alignment_layout.update_lines()

    that.frame_create = function (elem) {

        const drag = d3.drag()
            .on("start", startDragging)
            .on("drag", dragCircle)
            .on("end", endDragging);

        let lastNearest = null

        function dragCircle(ev, d) {
            let align_rect = d3.select(".card-upper-background").select('rect.align')
            let align_bg = d3.select(".card-lower-background")
            let dx = ev.x - d.start_ev.x
            let dy = ev.y - d.start_ev.y
            d.x = Math.max(range[0], Math.min(range[1], d.start.x + dx))
            d.y = d.start.y + dy

            let min_dist = 1e10
            let nearest = null
            
            let related_frames =
                that.cards[0].actions.filter(e => e.action_id == d.action_id)[0]
                .frames.filter(e => e.frame_idx != d.frame_idx && e.is_action >= 0)
            
            related_frames.forEach(e => {
                if (e.frame_idx < d.frame_idx && e.x > d.x || e.frame_idx > d.frame_idx && e.x < d.x) {
                    e.is_hidden = 1
                } else {
                    e.is_hidden = 0
                }
            })
            // console.log("related_frames", related_frames.map(e => e.is_hidden))

            for (let action of that.cards[0].actions) {
                if (!that.cards[0].normal_layout) {
                    if (action.cluster_idx > 0) continue
                }
                for (let frame of action.frames) {
                    if (frame.id == d.id) continue
                    if (frame.is_action < 0) continue
                    let dist = 0
                    if (that.cards[0].normal_layout) {
                        dist = (frame.x - d.x) * (frame.x - d.x) + (frame.y - d.y) * (frame.y - d.y)
                    } else {
                        dist = (frame.x - d.x) * (frame.x - d.x)
                    }
                    dist = Math.sqrt(dist)
                    if (dist < min_dist) {
                        min_dist = dist
                        nearest = frame
                    }
                }
            }

            if (min_dist > that.drag_min_dist) {
                nearest = null
                lastNearest = null
                align_rect.style('opacity', 0)
                align_bg.select('.must-link').style('opacity', 0)
                align_bg.select('.must-not-link').style('opacity', 0)
            } else {
                if (!that.cards[0].normal_layout) {
                    if (d.x >= that.cards[0].must_link_area.x1 && d.x <= that.cards[0].must_link_area.x2 &&
                        d.y >= that.cards[0].must_link_area.y1 && d.y <= that.cards[0].must_link_area.y2) {
                            align_bg.select('.must-link').style('opacity', 1)
                            align_bg.select('.must-not-link').style('opacity', 0)
                        }
                    else if (d.x >= that.cards[0].must_not_link_area.x1 && d.x <= that.cards[0].must_not_link_area.x2 &&
                        d.y >= that.cards[0].must_not_link_area.y1 && d.y <= that.cards[0].must_not_link_area.y2) {
                            align_bg.select('.must-not-link').style('opacity', 1)
                            align_bg.select('.must-link').style('opacity', 0)
                        }
                }

                let padding = 20
                let height = that.drag_min_dist + padding * 2
                let y = nearest.y - height / 2
                
                if (!that.cards[0].normal_layout) {
                    height = Math.abs(d.y - nearest.y) + padding * 2
                    y = nearest.y - padding
                }

                if (lastNearest) {
                    align_rect
                        .attr('x', nearest.x - padding)
                        .attr('width', padding * 2)
                        .attr('y', y)
                        .attr('height', height)
                        .style('opacity', 1)
                } else {
                    align_rect
                        .attr('x', nearest.x - padding)
                        .attr('width', padding * 2)
                        .attr('y', y)
                        .attr('height', height)
                        .transition()
                        .duration(Global.QuickAnimation)
                        .style('opacity', 1)
                }
                lastNearest = nearest
            }

            d3.select(this).attr("transform", `translate(${d.x},${d.y})`)
            that.alignment_layout.update_lines()
            that.single_action_line_update(d.action_id)
        }

        let range = [-1e10, 1e10]

        function startDragging(ev, d) {
            let related_frames =
                that.cards[0].actions.filter(e => e.action_id == d.action_id)[0]
                .frames.filter(e => e.frame_idx != d.frame_idx && e.is_action > 0)
            
                
            if (that.cards[0].normal_layout) {
                range = [
                    Math.max(...related_frames.filter(e => e.x < d.x).map(e => e.x)),
                    Math.min(...related_frames.filter(e => e.x > d.x).map(e => e.x))
                ]
            } else {
                range = [-1e10, 1e10]
            }
            lastNearest = null

            d.start_ev = { x: ev.x, y: ev.y }
            d.start = { x: d.x, y: d.y }
            d.is_dragging = 1
            d3.select(this).raise()
        }

        function endDragging(ev, d) {
            let align_rect = d3.select(".card-upper-background").select('rect.align')
            let align_bg = d3.select(".card-lower-background")
            let dx = ev.x - d.start_ev.x
            let dy = ev.y - d.start_ev.y
            d.x = d.start.x + dx
            d.y = d.start.y + dy
            d.is_dragging = 0
            align_rect.style('opacity', 0)
            align_bg.select('.must-link').style('opacity', 0)
            align_bg.select('.must-not-link').style('opacity', 0)

            d3.select(this)
                .select("frame-glyph")
                .transition()
                .duration(Global.QuickAnimation)
                .attr("transform", 'scale(1)')

            if (lastNearest) {
                d.x = lastNearest.x
                d3.select(this)
                    .transition()
                    .duration(Global.QuickAnimation)
                    .attr("transform", `translate(${d.x},${d.y})`)
                if (!that.cards[0].normal_layout) {
                    if (d.x >= that.cards[0].must_link_area.x1 && d.x <= that.cards[0].must_link_area.x2 &&
                        d.y >= that.cards[0].must_link_area.y1 && d.y <= that.cards[0].must_link_area.y2) {
                            that.operation['must-link'].push([d.id, lastNearest.id])
                        }
                    else if (d.x >= that.cards[0].must_not_link_area.x1 && d.x <= that.cards[0].must_not_link_area.x2 &&
                        d.y >= that.cards[0].must_not_link_area.y1 && d.y <= that.cards[0].must_not_link_area.y2) {
                            that.operation['must-not-link'].push([d.id, lastNearest.id])
                        }
                } else {
                    that.operation['must-link'].push([d.id, lastNearest.id])
                }
                console.log(that.operation)
            } else {
                if (d.x == d.start.x && d.y == d.start.y) {
                    if (d.is_action <= 0) return
                    if (d.is_selected) {
                        that.cards[0].rep_frames.forEach(e => {
                            if (e.cluster_idx == d.cluster_idx && e.frame_id == d.fid) {
                                e.frames.forEach((f) => {
                                    f.is_selected = 0
                                })
                                e.select_image_index = -1
                                e.show_image = 0
                            }
                        })
                    } else {
                        that.cards[0].rep_frames.forEach(e => {
                            if (e.cluster_idx == d.cluster_idx && e.fid == d.fid) {
                                e.frames.forEach((f) => {
                                    f.is_selected = 0
                                })
                                e.select_image_index = d.index
                                e.show_image = 1
                            }
                        })
                        d.is_selected = 1
                    }
                    that.frame_band_subcomponent_update(that.aligns_group, that.cards[0].rep_frames);
                    that.frame_point_subcomponent_update(that.aligns_group, that.cards[0].clusters);
                } else {
                    d.x = d.start.x
                    d.y = d.start.y
                    d3.select(this)
                        .transition()
                        .duration(Global.QuickAnimation)
                        .attr("transform", `translate(${d.x},${d.y})`)
                }
            }

            that.alignment_layout.update_lines()
            that.single_action_line_update(d.action_id, 1)
        }

        function frameInteraction(frame) {
            frame.on("mouseenter", function (ev, d) {
                if(that.parent.zoom_in || that.parent.change_rep) that.cluster_highlight(that.data.clusters[d.cluster_idx].cluster_id);
                if (d.is_action <= -1) return
                d3.select(this)
                    .raise()
                d3.select(this)
                    .transition()
                    .duration(Global.QuickAnimation)
                    .attr("transform", 'scale(2)')
            }).on("mouseout", function (ev, d) {
                if (d.is_action <= -1) return
                if (d.is_dragging) return
                d3.select(this)
                    .transition()
                    .duration(Global.QuickAnimation)
                    .attr("transform", 'scale(1)')
            })
        }

        let create = elem
            .enter()
            .filter(d => d.is_action >= 0)
            .append("g")
            .attr("class", "frame-group")
            .attr("id", d => "fg-" + d.id)
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
            .style("opacity", 0)
            .call(drag)

        create
            .transition()
            .duration(that.create_ani)
            .delay(that.update_ani + that.remove_ani)
            .style("opacity", 1)

        create.filter(d => d.is_action > -1 && d.is_selected)
            .raise()

        let glyph = create
            .append("g")
            .attr("class", "frame-glyph")
            .attr("transform", 'scale(1)')
            .call(frameInteraction)

        glyph
            .filter(d => d.is_single_frame)
            .append("path")
            .attr("class", "frame-star")
            .attr("d", "M14.1,0.43l3.44,8.05l8.72,0.78c0.39,0.03,0.67,0.37,0.64,0.76c-0.02,0.19-0.1,0.35-0.24,0.47l0,0 l-6.6,5.76l1.95,8.54c0.09,0.38-0.15,0.75-0.53,0.84c-0.19,0.04-0.39,0-0.54-0.1l-7.5-4.48l-7.52,4.5 c-0.33,0.2-0.76,0.09-0.96-0.24c-0.1-0.16-0.12-0.35-0.08-0.52h0l1.95-8.54l-6.6-5.76c-0.29-0.25-0.32-0.7-0.07-0.99 C0.3,9.35,0.48,9.28,0.66,9.27l8.7-0.78l3.44-8.06c0.15-0.36,0.56-0.52,0.92-0.37C13.9,0.13,14.03,0.27,14.1,0.43L14.1,0.43 L14.1,0.43z")
            .attr("transform", "scale(0.55) translate(-13, -15)")
            .style("fill", d => {
                if (d.is_action > -1 && d.is_selected) {
                    return Global.Orange
                } else {
                    if (d.is_action < 1) {
                        return Global.Blue40
                    } else {
                        return Global.MidBlue
                    }
                }
            })
            .style("stroke", "white")
            .style("stroke-width", 3)
            .style("opacity", 1)

        glyph
            .filter(d => !d.is_single_frame)
            .append("circle")
            .attr("class", "frame-circle")
            .attr("cx", d => 0)
            .attr("cy", d => 0)
            .attr("r", d => d.is_action > -1 ? 5 : 3.5)
            .style("display", d => d.is_action > -1 ? "block" : "none")
            .style("fill", d => {
                if (d.is_single_frame) {
                    return "red"
                }
                if (d.is_action > -1 && d.is_selected) {
                    return Global.Orange
                } else {
                    return Global.MidBlue
                    // if (d.is_action < 1) {
                    //     return Global.Blue40
                    // } else {
                    //     return Global.MidBlue
                    // }
                }
            })
            .style("stroke", "white")
            .style("stroke-width", d => {
                if (d.is_rep) {
                    return 1
                } else {
                    if (d.is_action <= -1) {
                        return 0
                    } else {
                        return 1
                    }
                }
            })
            .style("opacity", 1)

    };
    that.frame_update = function (elem) {
        let frame = elem
            .filter(d => d.show_band == 1)

        let single_frame = frame
            .filter(d => d.is_single_frame)
            .select(".frame-star")

        let is_zoom_in = false
        if (that.last_card && that.cards[0].depth - 1 == that.last_card.depth) {
            is_zoom_in = true
        }

        let update_delay
        // console.log('is_zoom_in', is_zoom_in)
        if (is_zoom_in) {
            update_delay = that.remove_ani
        } else {
            update_delay = that.remove_ani * 2 + Global.QuickAnimation
        }
        

        single_frame
            .style("fill", d => {
                if (d.is_action > -1 && d.is_selected) {
                    return Global.Orange
                } else {
                    if (d.is_action < 1) {
                        return Global.Blue40
                    } else {
                        return Global.MidBlue
                    }
                }
            })
            .style("stroke", "white")
            .style("stroke-width", 3)
            .style("opacity", 1)

        let common_frame = frame
            .filter(d => !d.is_single_frame)
            .select(".frame-circle")

        // console.log('common_frame', common_frame)
        common_frame
            .attr("r", d => d.is_action > -1 ? 5 : 3.5)
            .style("fill", d => {
                if (d.is_action > -1 && d.is_selected) {
                    return Global.Orange
                } else {
                    return Global.MidBlue
                    // if (d.is_action < 1) {
                    //     return Global.Blue40
                    // } else {
                    //     return Global.MidBlue
                    // }
                }
            })
            .style("stroke", "white")
            .style("stroke-width", d => {
                if (d.is_rep) {
                    return 1
                } else {
                    if (d.is_action <= -1) {
                        return 0
                    } else {
                        return 1
                    }
                }
            })
            .style("opacity", 1)

        single_frame.raise()
        // console.log('elem.filter(d => d.is_action > -1 && d.is_selected)', elem)
        elem.filter(d => d.is_action > -1 && d.is_selected)
            .raise()

        elem.transition()
            .duration(that.update_ani)
            .delay(update_delay)
            .attr("transform", d => `translate(${d.x}, ${d.y})`)
            .style("display", d => d.is_hidden ? "none" : "block")

    };

    that.frame_remove = function (elem) {
        elem
            .exit()
            .transition()
            .duration(that.remove_ani)
            .style("opacity", 0)
            .remove();
        
        elem
            .filter(d => d.is_action == -1)
            .style("opacity", 0)
            .remove();
    };

    that.layer_delta = 40;
    that.layer_width = 75;
    that.layer_height = 30;
    that.layer_dx = that.layer_height / 3;
    that.layer_dy = that.layer_height / 2;
    that.layer_r = that.layer_dy;
    that.home_width = 40;
    that.breadcrumb_path = function (x, y, compare = false) {
        let layer_width = compare ? that.layer_width + 25 : that.layer_width;
        return `M ${x},${y} l ${-that.layer_dx},${-that.layer_dy} l ${layer_width},${0} q ${that.layer_r},${0} ${that.layer_r},${that.layer_r}` +
            `q ${0},${that.layer_r} ${-that.layer_r},${that.layer_r} l ${-layer_width},${0} l ${that.layer_dx},${-that.layer_dy}`;
    }
    that.y_margin = -15;
    that.breadcrumbs_layout = function(){
        that.breadcrumbs = [{
            index: 0,
            x: 20,
            y: that.y_margin,
            path: that.breadcrumb_path(20, that.y_margin),
            type: "normal" 
        }]
        that.parent.history.forEach((value, index) => {
            index = index + 1
            let x0 = 20 + index * that.layer_delta;
            let y0 = that.y_margin;
            that.breadcrumbs.push({
                index: index,
                x: x0,
                y: y0,
                path: that.breadcrumb_path(x0, y0, value.is_comparison),
                type: value.is_comparison ? "comparison" : "normal"
            })
        });
    }
    that.breadcrumbs_create = function (elem) {
        let create = elem.enter()
            .append("g")
            .attr("class", "breadcrumb-group")
            .attr("id", d => `bc-${d.index}`)
            .attr("transform", d => `translate(${d.x}, ${d.y})`);
        create.style("opacity", 0)
            .transition()
            .duration(that.create_ani)
            .delay(that.update_ani + that.remove_ani)
            .style("opacity", 1);
        create.filter(d => d.index === 0)
            .append("rect")
            .attr("class", "bread-home-hull")
            .attr("x", d => d.x - that.home_width)
            .attr("y", d => d.y - that.layer_dy)
            .attr("width", 80)
            .attr("height", that.layer_height)
            .attr("rx", 10)
            .attr("ry", 10)
            .attr("stroke", "rgb(218,218,218)")
            .attr("stroke-width", 2)
            .attr("fill", "white");
        create.filter(d => d.index === 0)
            .append("g")
            .attr("transform", d => `translate(${d.x - that.home_width + 8}, ${d.y - 12})`)
            .append("svg")
            .attr("class", "bread-home-svg")
            .attr("viewBox", "0 0 1024 1024")
            .attr("width", 24)
            .attr("height", 24)
            .append("path")
            .attr("d", "M880.702 563.34c-2.501 3.002-6.503 5.004-10.506 5.504-0.5 0-1 0-1.501 0-4.002 0-7.504-1-10.505-3.502L512 276.683 165.81 565.342c-3.502 2.502-7.504 4.002-12.007 3.502-4.002-0.5-8.004-2.502-10.506-5.504l-31.017-37.02c-5.503-6.504-4.502-17.009 2.001-22.512l359.697-299.665c21.012-17.51 55.03-17.51 76.042 0L672.088 306.2l0-97.554c0-9.005 7.004-16.009 16.009-16.009l96.053 0c9.005 0 16.009 7.004 16.009 16.009l0 204.112 109.561 91.05c6.503 5.503 7.504 16.008 2.001 22.512L880.702 563.34zM800.158 800.971c0 17.51-14.508 32.018-32.018 32.018L576.035 832.989 576.035 640.883l-128.07 0 0 192.105L255.859 832.988c-17.51 0-32.018-14.508-32.018-32.018L223.841 560.84c0-1.002 0.5-2.002 0.5-3.002L512 320.708l287.658 237.13c0.5 1 0.5 2 0.5 3.002L800.158 800.971z")
            .attr("fill", "rgb(117,117,117)");

        create.append("path")
            .attr("class", "bread-hull")
            .attr("d", d => d.path)
            .attr("stroke", "rgb(218,218,218)")
            .attr("stroke-width", 2)
            .attr("fill", "white");
        // .attr("fill", d => (d.index === that.stack.length - 1) ? "rgb(158,189,62)" : "white");
        // .style("opacity", 0.2)
        create.append("text")
            .attr("class", "level-text")
            .attr("x", d => d.type === "normal" ? d.x + 10 : d.x + 30)
            .attr("y", d => d.y + 5)
            .style("fill", d => (d.index === that.breadcrumbs.length - 1) ? "black" : "rgb(117,117,117)")
            .text(d => d.type === "normal" ? `Level ${d.index + 1}` : "Select")
            .on("click", function (ev, d) {
                if (!that.data.is_anchor_alignment && d.index === that.stack.length - 1) return;
                that.stack = that.stack.slice(0, d.index + 1);
                that.parent.pop_history(d.index);
                that.data = that.stack[d.index];
                that.sub_component_view_update();
            });
        return create;
    }
    that.breadcrumbs_update = function (elem) {
        elem.select("text.level-text")
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .style("fill", d => (d.index === that.breadcrumbs.length - 1) ? "black" : "rgb(117,117,117)")
            .text(d => d.type === "normal" ? `Level ${d.index + 1}` : "Select")
    }
    that.breadcrumbs_remove = function (elem) {
        elem.exit()
            .transition()
            .duration(that.remove_ani)
            .style("opacity", 0)
            .remove();
    }


    // highlight functions
    that.cluster_highlight = function(cluster_id) {
        if(that.highlight_cluster_id !== cluster_id && that.highlight_cluster_id) 
            that.cluster_dehighlight(that.highlight_cluster_id);
        d3.select(`path.cluster-hull#cl-h-${cluster_id}`)
            .style('stroke', Global.Orange)
            .style('stroke-opacity', 1)
            // .style('fill', 'rgba(1,103,243,0.15)');
        that.highlight_cluster_id = cluster_id;
    }

    that.cluster_dehighlight = function(cluster_id) {
        d3.select(`path.cluster-hull#cl-h-${cluster_id}`)
            .style('stroke', "none")
            .style('stroke-opacity',0.1)
            // .style('fill', 'rgba(1,103,243,0.1)');
    }

    // video control functions
    that.play_video_from_fid = function(fid) {
        if(that.video.view.readyState < 4) return;
        that.play_interval && clearInterval(that.play_interval);
        that.video.view.currentTime = that.video.fid_to_t(fid);
        let start = that.selected_action_data.findIndex(d => d.frame_id === fid);
        let end = that.selected_action_data.length - 1;
        let cur_x = that.selected_action_data[start].x + that.selected_width * 3 / 2;
        d3.select("line#selected-action-video-pos")
            .interrupt()
            .attr("x1", cur_x)
            .attr("x2", cur_x)
        d3.select(`#selected-action-frame-${fid}`).raise()
        let move = function(){
            if(start === end){
                clearInterval(that.play_interval);
                if(!that.video.view.paused) that.video.view.pause();
                // d3.select('.selected-action-frame-info').raise()
                return;
            }
            start += 1; fid += 1;
            cur_x = that.selected_action_data[start].x + that.selected_width * 3 / 2;
            d3.select("line#selected-action-video-pos")
                .transition()
                .duration(that.video.delta)
                .ease(d3.easeLinear)
                .attr("x1", cur_x)
                .attr("x2", cur_x)
            // d3.select(`#selected-action-frame-${fid}`).raise()
        }
        if(that.video.view.paused) that.video.view.play();
        that.play_interval = setInterval(move, that.video.delta);
    }

    that.pause_video_to_fid = function(fid) {
        that.play_interval && clearInterval(that.play_interval);
        if(!that.video.view.paused) that.video.view.pause();
        that.video.view.currentTime = that.video.fid_to_t(fid);
        let start = that.selected_action_data.findIndex(d => d.frame_id === fid);
        let cur_x = that.selected_action_data[start].x + that.selected_width * 3 / 2;
        d3.select("line#selected-action-video-pos")
            .interrupt()
            .attr("x1", cur_x)
            .attr("x2", cur_x)
        // d3.select(`#selected-action-frame-${fid}`).raise()
    }

    // star labeled frames
    that.star_labeled_frames = function(){
        if (that.cards[0].normal_layout) {
            that.cards[0].rep_frames.forEach(e => {
                if(!e.show_band) return;
                let flag = 0
                e.frames.forEach((f, index) => {
                    if (f.is_action > 0 && f.is_single_frame && flag === 0) {
                        e.select_image_index = index
                        f.is_selected = 1
                        flag = 1
                    }
                })
                if (flag) {
                    e.frames.forEach((f, index) => {
                        if (index !== e.select_image_index) {
                            f.is_selected = 0
                        }
                    })
                }
            })
            that.frame_band_subcomponent_update(that.aligns_group, that.cards[0].rep_frames);
            that.frame_point_subcomponent_update(that.aligns_group, that.cards[0].clusters);
        }
    }
}

export default AlignMentsRender; 