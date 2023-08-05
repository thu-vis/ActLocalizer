import * as d3 from 'd3';
import * as Global from './global';

const VideoInfoRender = function (parent, ratios) {
    // ------------------------ render args and data ------------------------
    let that = this;
    that.parent = parent;
    that.ratios = ratios;

    that.args = {
        colors: {
            handle: 'darkgrey', 
            bar: d3.schemeDark2[5],
            frame: 'teal'
        },
        timeline: {
            x_margin: 10,
            y_top_margin: 20,
            axis_height: 20,
            
        },
        localline: {
            x_margin: 60,
            y_top_margin: 25,
            x_in_margin: 10,
            y_in_margin: 24
        },
        keyframes: {
            x_margin: 5,
            y_bottom_margin: 10,
            frame_width: 70,
            frame_height: 70
        }
    };

    that.data = {
        timeline: [],
        key_frames: [],
        selected: {
            bound: [],
            keys: []
        },
    };

    that.layout = {
        timeline: {},
        localline: {},
        keyframes: {}
    }

    // ------------------------ render initialization ------------------------
    
    that.update_info_from_parent = function(){
        // canvas
        that.video_group = that.parent.video_group;

        // layout
        that.layout_width = that.parent.layout_width;
        that.layout_height = that.parent.layout_height;
        that.timeline_width = that.layout_width - 2 * that.args.timeline.x_margin;
        that.timeline_height = that.layout_height * that.ratios[0]
                        - that.args.timeline.y_top_margin - that.args.timeline.axis_height;
        that.localline_width = that.layout_width - 2 * that.args.localline.x_margin;
        that.localline_height = that.layout_height * that.ratios[1] - that.args.localline.y_top_margin;
        that.keyframes_width = that.layout_width - 2 * that.args.keyframes.x_margin;
        that.keyframes_height = that.args.keyframes.frame_height

        // animation
        that.create_ani = that.parent.create_ani;
        that.update_ani = that.parent.update_ani;
        that.remove_ani = that.parent.remove_ani;
    }

    that.update_info_from_parent();

    that.timeline_group = that.video_group.append('g')
        .attr('id', 'timeline-group')
        .attr('transform', `translate(${that.args.timeline.x_margin}, ${that.args.timeline.y_top_margin})`);

    that.localline_group = that.video_group.append('g')
        .attr('id', 'localline-group')
        .attr('transform', `translate(${that.args.localline.x_margin}, 
            ${that.layout_height * that.ratios[0] + that.args.localline.y_top_margin})`);
    
    that.keyframes_group = that.video_group.append('g')
        .attr('id', 'keyframes-group')
        .attr('transform', `translate(${that.args.keyframes.x_margin}, 
            ${that.layout_height - that.keyframes_height - that.args.keyframes.y_bottom_margin})`);

    // ------------------------ update interactive ------------------------

    that.sub_component_update = async function (video, action) {
        // update info
        that.update_info_from_parent();

        // update state
        that.video = video;
        that.action = action;
        that.data.timeline = d3.range(...[0, video.length])
            .map(d => {
                return {
                    index: d,
                    value: video.scores[d]
                }
            });
        that.data.key_frames = [];
        that.data.selected.bound = [action.left_bound, action.right_bound]

        // update view
        that.timeline_component_update();
        await that.localline_component_update(true);
        that.keyframes_component_update();
    }
    
    that.update_key_frames_data = async function (bound) {
        // calculate target
        let target = Math.ceil((bound[1] + 1- bound[0]) / 3)
        if(target > 5) target = 5;

        let key = {"id": parseInt(that.parent.selected_action.id.split("-")[0]),
                    "bound": [bound[0], bound[1]],
                    "target": target}
        await that.parent.fetch_key_frames(key);
        that.data.key_frames = that.parent.key_frames.slice(0, 5).map(d => d + bound[0]);
    }

    // ------------------------ functions for render ------------------------
    // 1. timeline comnponent
    that.timeline_component_update = function() {
        that.timeline_layout();
        that.timeline_create();
        that.timeline_update();
        that.timeline_remove();
        that.setbrush();
    }

    that.timeline_layout = function() {
        let data = that.data.timeline

        let xScale = d3.scaleBand()
            .domain(data.map(d => d.index))
            .range([0, that.timeline_width]);
        let yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.value))
            .range([0, that.timeline_height]);
        let xAxis = d3.axisBottom()
            .scale(xScale)
            .tickValues(xScale.domain().filter((d, i) => !(i % 100)));

        let line = d3.line()
            .x(d => xScale(d.index) + 0.5 * xScale.bandwidth())
            .y(d => that.timeline_height - yScale(d.value))
            .curve(d3.curveNatural);

        that.layout.timeline.xScale = xScale;
        that.layout.timeline.yScale = yScale;
        that.layout.timeline.axis = xAxis;
        that.layout.timeline.line = line;
    }

    that.timeline_create = function() {
        // console.log('timeline_create');

        let data = that.data.timeline;
        let xAxis = that.layout.timeline.axis;
        let g = that.timeline_group;

        // Axis
        g.selectAll('.axis')
            .data(['x'])
            .enter()
            .append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0, ${that.timeline_height + 5})`)
            .call(xAxis)

        // Bars
        // that.layout.timeline.bar = g.selectAll('rect.bar')
        //     .data(data);
        // that.layout.timeline.bar
        //     .enter()
        //     .append('rect')
        //     .attr('class', 'bar')
        //     .attr('id', d => 'idx-' + d.index)
        //     .attr('x', d => xScale(d.index))
        //     .attr('y', d => that.timeline_height - yScale(d.value))
        //     .attr('width', xScale.bandwidth())
        //     .attr('height', d => yScale(d.value))
        //     .attr('fill', colors.bar)
        //     .attr('opacity', 1);

        // line
        g.selectAll('path.line')
            .data(['total_line'])
            .enter()
            .append('path')
            .attr('class', 'line')
            .attr('d', that.layout.timeline.line(data))
            .style('stroke', that.args.colors.bar)
            .style('stroke-width', 1)
            .style('fill', 'transparent');
    }

    that.timeline_update = function() {
        let xAxis = that.layout.timeline.axis;
        let g = that.timeline_group;

        // Axis
        g.selectAll('.axis')
            .call(xAxis)

        // Bars
        // that.layout.timeline.bar
        //     .attr('x', d => xScale(d.index))
        //     .attr('y', d => that.timeline_height - yScale(d.value))
        //     .attr('width', xScale.bandwidth())
        //     .attr('height', d => yScale(d.value))

        // line
        g.selectAll('path.line')
            .attr('d', that.layout.timeline.line(that.data.timeline));

    }

    that.timeline_remove = function() {
        // console.log('timeline_remove', that.layout.timeline.bar);
        // that.layout.timeline.bar
        //     .exit()
        //     .style('opacity', 0)
        //     .remove();
    }

    that.setbrush = function() {
        let snappedSelection = function (bandScale, domain) {
            const min = d3.min(domain),
                max = d3.max(domain);
            return [bandScale(min), bandScale(max) + bandScale.bandwidth()]
        };

        let filteredDomain = function (scale, min, max) {
            let dif = scale(d3.min(scale.domain())) - scale.range()[0],
                iMin = (min - dif) < 0 ? 0 : Math.round((min - dif) / scale.step()),
                iMax = Math.round((max - dif) / scale.step());
            if (iMax == iMin) --iMin; 

            return scale.domain().slice(iMin, iMax)
        };

        let brushing = function (xScale) {
            return async function (event) {
                if (!event.selection && !event.sourceEvent) return;
                const s0 = event.selection ? event.selection : [1, 2].fill(event.sourceEvent.offsetX),
                    d0 = filteredDomain(xScale, ...s0);
                let s1 = s0;

                if (event.sourceEvent && event.type === 'end') {
                    s1 = snappedSelection(xScale, d0);
                    d3.select(this).transition().call(event.target.move, s1);
                }

                if(that.data.selected.bound[0] === d0[0] &&
                    that.data.selected.bound[1] === d0[d0.length - 1])
                    return;
                
                that.data.selected.bound = [d0[0], d0[d0.length - 1]];
                that.layout.timeline.handle_pos = snappedSelection(xScale, that.data.selected.bound);

                // move handlers
                d3.selectAll('g.handles')
                    .attr('transform', d => {
                        const x = d == 'handle--o' ? s1[0] : s1[1];
                        return `translate(${x}, 0)`;
                    });

                // update labels
                d3.selectAll('g.handles').selectAll('text')
                    .attr('dx', d0.length > 1 ? 0 : 6)
                    .text((d, i) => {
                        let index;
                        if (d0.length > 1) {
                            index = d == 'handle--o' ? d3.min(d0) : d3.max(d0);
                        } else {
                            index = d == 'handle--o' ? d3.min(d0) : '';
                        }
                        return index;
                    })

                // // update bars
                // that.timeline_group.selectAll('.bar')
                //     .attr('opacity', d => d0.includes(d.index) ? 1 : 0.2);

                // update localline
                await that.localline_component_update();
                that.keyframes_component_update();
            }
        };

        let clearingframes = function(){
            that.data.key_frames = []
            that.layout.localline.key_index = []
            that.keyframes_component_update();
        }

        let updatingframes = async function(){
            console.log("change_bound", bound = that.data.selected.bound)
            await that.localline_component_update(true);
            that.keyframes_component_update();
        }

        let xScale = that.layout.timeline.xScale;

        let brush = d3.brushX()
            .handleSize(8)
            .extent([[0, 0], [that.timeline_width, that.timeline_height]])
            .on('start brush end', brushing(xScale))
            .on('start .rect', clearingframes)
            .on('end .rect', updatingframes);
        
        that.layout.timeline.handle_pos = snappedSelection(xScale, that.data.selected.bound)

        let colors = that.args.colors;
        let g = that.timeline_group;
        let bound = that.data.selected.bound,
            pos = that.layout.timeline.handle_pos;

        // create/update brush and handle
        g.selectAll('.brush')
            .data(['brush'])
            .enter()
            .append('g')
            .attr('class', 'brush');

        let gBrush = g.selectAll('.brush')
            .call(brush)
            .call(brush.move, [pos[0], pos[1]]);

        gBrush.selectAll('g.handles')
            .data(['handle--o', 'handle--e'])
            .enter()
            .append('g')
            .attr('class', d => `handles ${d}`)
            .attr('fill', colors.handle);

        let gHandles = g.selectAll('g.handles')
            .attr('transform', d => {
                const x = d == 'handle--o' ? pos[0] : pos[1];
                return `translate(${x}, 0)`;
            });

        // Label
        gHandles.selectAll('text')
            .data(d => [d])
            .enter()
            .append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', -10)
            .attr('font-size', 10)
            .text(d => d == 'handle--o' ? bound[0] : bound[1]);

        // Visible Line
        gHandles.selectAll('.line')
            .data(d => [d])
            .enter()
            .append('line')
            .attr('class', d => `line ${d}`)
            .attr('x1', 0)
            .attr('y1', -5)
            .attr('x2', 0)
            .attr('y2', that.timeline_height + 5)
            .attr('stroke', colors.handle);
    }

    // 2. localline comnponent
    that.localline_component_update = async function(update_key_frames=false) {
        await that.localline_layout(update_key_frames);
        that.localline_create();
        that.localline_update();
        that.links_tl_to_ll();
    }

    that.localline_layout = async function(update_key_frames) {
        // line chart axis
        let linechart_width = that.localline_width - 2 * that.args.localline.x_in_margin;
        let linechart_height = that.localline_height - 2 * that.args.localline.y_in_margin;

        let bound = that.data.selected.bound,
            data = that.data.timeline.slice(bound[0], bound[1] + 1);

        let xScale = d3.scaleLinear()
            .domain([bound[0], bound[1]])
            .range([0, linechart_width]);
        let yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.value)])
            .range([linechart_height, 0]);

        let ticks = [bound[0]], 
            band = Math.pow(10, Math.floor(Math.log10(bound[1] - bound[0] + 2)));
        for(let i = bound[0] + 1;i < bound[1];i++){
            if(i % band == 0) ticks.push(i);
        }
        ticks.push(bound[1]);
        let xAxis = d3.axisBottom()
            .scale(xScale)
            .tickValues(ticks);

        let line = d3.line()
            .x(d => xScale(d.index))
            .y(d => yScale(d.value))
            .curve(d3.curveNatural);
        
        that.layout.localline.data = data
        that.layout.localline.line = line
        that.layout.localline.axis = xAxis

        if(update_key_frames)
            await that.update_key_frames_data(bound);
        that.layout.localline.key_index = 
            that.data.key_frames.filter(d => (d - bound[0]) * (bound[1] - d) >= 0);
        that.layout.localline.key_pos =
            that.layout.localline.key_index.map(d => xScale(d))
    }

    that.localline_create = function() {
        // console.log('localline_create');

        // draw line chart 
        that.localline_group.selectAll('path.line')
            .data(['local_line'])
            .enter()
            .append('path')
            .attr('class', 'line')
            .attr('d', that.layout.localline.line(that.layout.localline.data))
            .style('stroke', that.args.colors.bar)
            .style('stroke-width', 1)
            .attr('transform', `translate(${that.args.localline.x_in_margin}, 
                    ${that.args.localline.y_in_margin})`)
            .style('fill', 'none');
        
        // x axis info
        that.localline_group.selectAll('.axis')
            .data(['x'])
            .enter()
            .append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(${that.args.localline.x_in_margin}, 
                    ${that.localline_height - that.args.localline.y_in_margin + 5})`)
            .call(that.layout.localline.axis)
    }

    that.localline_update = function() {
        that.localline_group.selectAll('path.line')
            .attr('d', that.layout.localline.line(that.layout.localline.data));

        that.localline_group.selectAll('.axis')
            .call(that.layout.localline.axis)
    }

    that.links_tl_to_ll = function() {
        let g = that.video_group;

        // left and right links
        let left = {
            x1: that.args.timeline.x_margin + that.layout.timeline.handle_pos[0],
            y1: that.args.timeline.y_top_margin + that.timeline_height + 5,
            x2: that.args.localline.x_margin,
            y2: that.layout_height * that.ratios[0] + that.args.localline.y_top_margin,
        }, right = {
            x1: that.args.timeline.x_margin + that.layout.timeline.handle_pos[1],
            y1: that.args.timeline.y_top_margin + that.timeline_height + 5,
            x2: that.args.localline.x_margin + that.localline_width,
            y2: that.layout_height * that.ratios[0] + that.args.localline.y_top_margin,
        };

        let links = g.selectAll('line.tltoll')
            .data([left, right]);
        links.enter()
            .append('line')
            .attr('class', 'tltoll')
            .attr('x1', d => d.x1)
            .attr('y1', d => d.y1)
            .attr('x2', d => d.x2)
            .attr('y2', d => d.y2)
            .attr('stroke', that.args.colors.handle);
        links.attr('x1', d => d.x1)
            .attr('y1', d => d.y1)
            .attr('x2', d => d.x2)
            .attr('y2', d => d.y2);

        // background square
        if (that.layout.localline.square === undefined) {
            that.layout.localline.square = [];
            that.layout.localline.square.push({
                x: that.args.localline.x_margin,
                y: that.layout_height * that.ratios[0] + that.args.localline.y_top_margin
            });
            that.layout.localline.square.push({
                x: that.args.localline.x_margin + that.localline_width,
                y: that.layout_height * that.ratios[0] + that.args.localline.y_top_margin
            });
            that.layout.localline.square.push({
                x: that.args.localline.x_margin + that.localline_width,
                y: that.layout_height * that.ratios[0] + that.args.localline.y_top_margin
                + that.localline_height
            });
            that.layout.localline.square.push({
                x: that.args.localline.x_margin,
                y: that.layout_height * that.ratios[0] + that.args.localline.y_top_margin
                + that.localline_height
            });
            that.layout.localline.square.push({
                x: that.args.localline.x_margin,
                y: that.layout_height * that.ratios[0] + that.args.localline.y_top_margin
            });
        }
        g.selectAll('path.llsquare')
            .data(['square'])
            .enter()
            .append('path')
            .attr('class', 'llsquare')
            .attr('d', d3.line()
                .x(d => d.x)
                .y(d => d.y)(that.layout.localline.square))
            .attr('fill', 'none')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', 8 + ' ' + 4)
            .attr('stroke', that.args.colors.handle);
    }

    // 3. keyframes component
    that.keyframes_component_update = function() {
        that.keyframes_layout();
        that.keyframes_create();
        that.keyframes_update();
        that.keyframes_remove();
        that.links_ll_to_kf();
    }

    that.keyframes_layout = function() {
        let num = that.layout.localline.key_index.length;
        let gap = (that.keyframes_width - num * that.args.keyframes.frame_width)
            / (num + 1);
        
        let frames = [];
        for(let i = 0; i < num;i++){
            let frame = {}
            frame.index = that.layout.localline.key_index[i];
            frame.x = (i + 1) *gap + i * that.args.keyframes.frame_width;
            frame.y = 0;
            frames.push(frame);
        }
        that.layout.keyframes.data = frames;
    }

    that.keyframes_create = function() {
        that.layout.keyframes.frames = 
            that.keyframes_group.selectAll('image.keyframe')
            .data(that.layout.keyframes.data);
            console.log('Frame/single_frame?filename', that.video.id, d.index)
        that.layout.keyframes.frames.enter()   
            .append('image')
            .attr('class', 'keyframe')
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('width', that.args.keyframes.frame_width)
            .attr('height', that.args.keyframes.frame_height)
            .attr('xlink:href', d => `${that.parent.server_url}/Frame/single_frame?filename=${that.video.id}-${d.index}`)
    }

    that.keyframes_update = function() {
        that.layout.keyframes.frames
            .transition()
            .duration(this.remove_ani)
            .attr('x', d => d.x);
    }

    that.keyframes_remove = function() {
        that.layout.keyframes.frames
            .exit()
            .attr('opacity', 0)
            .remove();
    }

    that.links_ll_to_kf = function() {
        let g = that.video_group;
        let num = that.layout.localline.key_index.length;

        let nodedata = [],
            linkdata = [];
        for(let i = 0;i < num;i++){
            let node = {}, 
                link = {};
            node.index = i;
            node.x = that.layout.localline.key_pos[i] + that.args.localline.x_margin
                + that.args.localline.x_in_margin;
            node.y = that.layout_height * (that.ratios[0] + that.ratios[1]) - 20;
            nodedata.push(node)

            link.index = i;
            link.x1 = node.x;
            link.y1 = node.y;
            link.x2 = that.args.keyframes.x_margin + that.layout.keyframes.data[i].x 
                + 0.5 * that.args.keyframes.frame_width;
            link.y2 = that.layout_height - that.args.keyframes.y_bottom_margin - that.keyframes_height;
            linkdata.push(link)
        }
        
        let nodes = g.selectAll('circle.lltokf')
            .data(nodedata);
        nodes.enter()
            .append('circle')
            .attr('class', 'lltokf')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', 3)
            .attr('fill', that.args.colors.frame);
        nodes.attr('cx', d => d.x)
        nodes.exit()
            .attr('opacity', 0)
            .remove();

        let links = g.selectAll('line.lltokf')
            .data(linkdata);
        links.enter()
            .append('line')
            .attr('class', 'lltokf')
            .attr('x1', d => d.x1)
            .attr('y1', d => d.y1)
            .attr('x2', d => d.x2)
            .attr('y2', d => d.y2)
            .attr('stroke', that.args.colors.frame);
        links.attr('x1', d => d.x1)
            .transition()
            .duration(this.remove_ani)
            .attr('x2', d => d.x2);
        links.exit()
            .attr('opacity', 0)
            .remove();
    }

}

export default VideoInfoRender; 