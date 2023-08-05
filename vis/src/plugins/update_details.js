import * as d3 from 'd3';
import * as Global from './global';
import * as d3ContextMenu from "d3-context-menu";

const generate_data_from_action = function(action, line_height){
    let detailed_frames = [];
    const rep_frames = new Set(action.rep_frames)
    for (let frame_id = action.bound[0]; frame_id <= action.bound[1]; ++frame_id) {
        const id = `${action.vid}-${frame_id}`
        detailed_frames.push({
            y: line_height + 30,
            rep: rep_frames.has(id),
            id: id,
            frame_id: frame_id,
            is_initial: frame_id >= action.initial_bound[0] 
                && frame_id <= action.initial_bound[1],
        })
    }
    return detailed_frames;
}


const DetailUpdate = function(parent){
    let that = this;

    that.update_info_from_parent = function(){
        // canvas
        that.aligns_group = that.parent.aligns_group;

        // layout
        that.alignment_layout = that.parent.alignment_layout;
        that.layout_width = that.parent.aligns_width;
        that.layout_height = that.parent.aligns_height;
        that.width = that.parent.cards[0].width;

        // state
        that.cards = that.parent.cards;
        that.selected_details = that.parent.selected_details;

        // animation
        that.create_ani = that.parent.create_ani;
        that.update_ani = that.parent.update_ani;
        that.remove_ani = that.parent.remove_ani;
    }

    that.update_info_from_parent();

    that.component_update = function(){
        that.update_info_from_parent();

        let data = that.selected_details.detailed_frames;
        let vid = that.selected_details.vid;
        let selected_class = that.selected_details.class;
        let bound = that.selected_details.bound;
        let groups = that.aligns_group;
        
        const frame_height = that.alignment_layout.large_frame_height
        const frame_width = that.alignment_layout.large_frame_width
        const margin = frame_width
        const width = that.width - margin * 2
        const image_margin = Math.min(frame_width + 5, width / data.length)
        const start_x = (width - data.length * image_margin) / 2
        for (let i = 0; i < data.length; ++i) {
            data[i].x = i * image_margin + start_x
        }

        let frames = groups
            .selectAll(".selected-action-frames-group")
            .data(data, d => d.id)

        groups
            .selectAll(".selected-action-frame-info")
            .remove()

        let g = frames.enter()
            .append('g')
            .attr('transform', d => `translate(${d.x + margin},${d.y})`)
            .attr('class', 'selected-action-frames-group')
            .style('opacity', 0)
        

        g.transition()
            .duration(this.create_ani)
            .style('opacity', 1)

        g.append('text')
            .attr("dx", frame_width / 2)
            .attr("dy", -20)
            .attr("font-weight", 650)
            .attr("text-anchor", "middle")
            .attr("fill", Global.DeepGray)
            .text(d => `${d.frame_id}`)

        g.append('image')
            .attr('width', frame_width)
            .attr('height', frame_height)
            .attr('xlink:href', d => `${that.parent.server_url}/Frame/single_frame?filename=${d.id}`)
            .on("contextmenu", d3ContextMenu([
                {
                  title: "Boundary",
                  action: function() {
                    console.log("Confirm!");
                  },
                }
              ]));

        frames.transition()
            .duration(this.update_ani)
            .attr('transform', d => `translate(${d.x + margin},${d.y})`)

        frames.exit()
            .attr('opacity', 0)
            .remove();

        let left_x = Math.min(...data.map(d => d.x))
        let right_x = Math.max(...data.map(d => d.x)) + frame_width * 2
        let y = data[0].y
        let left_initial_x = Math.min(...data.filter(d => d.is_initial).map(d => d.x)) + frame_width
        let right_initial_x = Math.max(...data.filter(d => d.is_initial).map(d => d.x)) + frame_width * 2

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

        that.parent.fetch_pred_scores_of_video_with_given_boundary({
            "id": vid,
            "class": selected_class,
            "bound": bound,
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

            groups
                .append("path")
                .attr("transform", `translate(0,${y + frame_height + that.alignment_layout.margin})`)
                .attr("class", "selected-action-frame-info")
                .attr("d", path2)
                .style("fill", "none")
                .style("stroke", Global.DarkBlueGray)
                .style("stroke-width", "3px")
                .style("opacity", 0)
                .transition()
                .duration(that.create_ani)
                .delay(that.update_ani)
                .style("opacity", 1)

            groups
                .append("path")
                .attr("transform", `translate(0,${y + frame_height + that.alignment_layout.margin})`)
                .attr("class", "selected-action-frame-info")
                .attr("d", path1)
                .style("fill", Global.Blue40)
                .style("fill-opacity", .5)
                .style("stroke", "none")
                .style("opacity", 0)
                .transition()
                .duration(that.create_ani)
                .delay(that.update_ani)
                .style("opacity", 1)

                    
            groups.append("text")
                .attr("class", "selected-action-frame-info")
                .attr("transform", `translate(${left_x + frame_width / 4},${y + frame_height * 1.25 + that.alignment_layout.single_area_height / 2})`)
                .attr("font-size", "16px")
                .text("Confidence")
        })

        groups.append("line")
            .attr("class", "selected-action-frame-info")
            .attr('x1', 0)
            .attr('x2', that.layout_width)
            .attr('y1', y - 60)
            .attr('y2', y - 60)
            .style('stroke', Global.DeepGray)
            .style('opacity', .4)
            .style('stroke-width', 1.5)

        groups.append("rect")
            .attr("class", "selected-action-frame-info")
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

        groups.append("path")
            .attr("class", "selected-action-frame-info")
            .attr("transform", `translate(${left_x + frame_width / 4},${y + frame_height / 4}) scale(0.075)`)
            .attr("d", "M731.0336 859.8528V164.1472c0-40.1408-48.5376-60.3136-77.0048-31.8464L306.176 480.1536c-17.6128 17.6128-17.6128 46.1824 0 63.7952l347.8528 347.8528c28.4672 28.3648 77.0048 8.2944 77.0048-31.9488z")
            .style("fill", Global.GrayColor)
            .style("stroke", "none")
            .style("opacity", .5)
            .on("mouseenter", function(){
                d3.select(this)
                    .transition().duration(Global.QuickAnimation)
                    .style("opacity", 1)
            })
            .on("mouseout", function(){
                d3.select(this)
                    .transition().duration(Global.QuickAnimation)
                    .style("opacity", .5)
            })
            .on("click", function(){
                // d.current_action.bound[0] -= 3
                // that.sub_component_view_refresh()
            })

        groups.append("path")
            .attr("class", "selected-action-frame-info")
            .attr("transform", `translate(${right_x + frame_width / 4},${y + frame_height / 4}) scale(0.075)`)
            .attr("d", "M292.9664 164.1472v695.808c0 40.1408 48.5376 60.3136 77.0048 31.8464L717.824 543.8464c17.6128-17.6128 17.6128-46.1824 0-63.7952L369.9712 132.1984c-28.4672-28.3648-77.0048-8.2944-77.0048 31.9488z")
            .style("fill", Global.GrayColor)
            .style("stroke", "none")
            .style("opacity", .5)
            .on("mouseenter", function(){
                d3.select(this)
                    .transition().duration(Global.QuickAnimation)
                    .style("opacity", 1)
            })
            .on("mouseout", function(){
                d3.select(this)
                    .transition().duration(Global.QuickAnimation)
                    .style("opacity", .5)
            })
            .on("click", function(){
                // d.current_action.bound[1] += 3
                // that.sub_component_view_refresh()
            })

    };
};


export default DetailUpdate; 