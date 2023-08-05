import * as d3 from "d3";
import { svg } from "d3";
import * as Global from "./global";

const LabelsRender = function (parent) {
    let that = this;
    that.parent = parent;
    that.line_height = 40;
    that.line_margin = 10;
    that.width_margin = 20;
    that.width_svg = 18;
    that.width_ratios = [0.2, 0.55, 0.25];
    that.cur_height = that.line_height + 2 * that.line_margin;
    that.font_size = 18;


    that.update_info_from_parent = function () {
        that.server_url = that.parent.server_url;
        that.labels_group = that.parent.labels_group;
        that.labels_data = Global.deepCopy(that.parent.class_info);
        that.labels_data.sort((a,b) => {
            return a.id > b.id;
        });
        that.current_class = that.parent.selected_class;

        // animation
        that.create_ani = that.parent.create_ani;
        that.update_ani = that.parent.update_ani;
        that.remove_ani = that.parent.remove_ani;
        that.layout_width = that.parent.layout_width;
        that.layout_height = that.parent.layout_height;

    }
    that.update_info_from_parent();

    that.sub_component_update = function () {
        // update info
        that.update_info_from_parent();

        // update state
        console.log(that.labels_data)
        let data = that.layout()

        // update view
        that.e_lines = that.labels_group
            .selectAll(".work-labels")
            .data(data);

        that.create();
        that.update();
        that.remove();

        return that.cur_height;
    }

    that.layout = function(){
        let data = that.labels_data;

        let widths = [], cur_width = 0;
        that.width_ratios.forEach(d => {
            cur_width += that.layout_width * d / 2;
            widths.push(cur_width);
            cur_width += that.layout_width * d / 2;
        })
        that.widths = widths;
        let ret = []
        ret.push({
            type: 'schema',
            id: -1,
            x: 0,
            y: that.line_margin * 2 / 3,
        })
        ret.push({
            type: 'data',
            id: -2,
            index: 0,
            full_name: 'Background',
            x: 0,
            y: that.line_margin * 2 / 3 + that.line_height + that.line_margin,
        })
        data.forEach((d, index) => {
            d.type = 'data',
            d.index = index + 1,
            d.x = 0,
            d.y = that.line_margin * 2 / 3 + (that.line_height + that.line_margin) * (index + 2)
            ret.push(d);
        })
        if(ret.length > 1) ret[ret.length - 1].type = 'final';
        that.cur_height = (that.line_height + that.line_margin) * ret.length;
        return ret;
    }

    that.create = function () {
        console.log("labels_create")
        let line_groups = that.e_lines
            .enter()
            .append("g")
            .attr("class", "work-labels")
            .attr("id", (d) => "work-labels" + d.id)
            .attr("transform", (d) => "translate(" + d.x + ", " + d.y + ")")
            .attr("fill", (d) => d.id == that.current_class ? Global.DarkOrange: "black");

        line_groups.append("text")
            .text(d => d.type === 'schema' ? 'Id' : `${d.index}`)
            .attr("text-anchor", "middle")
            .attr("font-size", that.font_size)
            .attr("font-weight", d => d.type === "schema" ? "bold" : "normal")
            .attr("opacity", 0.6)
            .attr("transform", `translate(${that.widths[0]}, ${(that.line_height - that.font_size) / 2})`);

        line_groups.append("text")
            .text(d => d.type === 'schema' ? 'ClassName' : d.full_name)
            .attr("text-anchor", "middle")
            .attr("font-size", that.font_size)
            .attr("font-weight", d => d.type === "schema" ? "bold" : "normal")
            .attr("opacity", 0.6)
            .attr("transform", `translate(${that.widths[1]}, ${(that.line_height - that.font_size) / 2})`);


        line_groups.filter(d => d.type !== 'final')
            .append("line")
            .attr("x1", that.width_margin)
            .attr("y1", that.line_height - (that.line_height - that.font_size) / 2)
            .attr("x2", that.layout_width - that.width_margin)
            .attr("y2", that.line_height - (that.line_height - that.font_size) / 2)
            .attr("stroke", Global.DeepGray)
            .attr("stroke-width", d => d.type === 'schema' ? 4 : 1)
            .attr("opacity", d => d.type === 'schema' ? 0.6 : 0.3)

        let svgs = line_groups.filter(d => d.type !== 'schema')
            .append("g")
            .attr("transform", `translate(${that.widths[2] - that.width_svg / 2}, ${(that.line_height - that.width_svg) / 2 - that.width_svg + 2})`)
            .append("svg")
            .attr("width", that.width_svg)
            .attr("height", that.width_svg)
            .attr("viewBox", "0 0 1024 1024");
        svgs.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 1024)
            .attr("height", 1024)
            .attr("opacity", 0)
        svgs.append("path")
            .attr("d", "M834.603419 956.481234c-28.690436 0.001023-56.695257-11.376095-77.19515-31.875988L547.818787 714.822358c-0.450255-0.449231-0.436952-0.396019-0.765433-0.802272-2.405793-2.443655-6.221702-4.738931-10.997472-2.26253-0.246617 0.127913-0.49528 0.249687-0.747014 0.36532-46.595224 21.562085-96.394421 32.495088-148.013055 32.495088-47.664579 0-93.912903-9.339716-137.461745-27.759227-42.053792-17.787108-79.817883-43.246967-112.243387-75.671447-32.425504-32.426527-57.885362-70.189595-75.67247-112.242364-18.419512-43.548842-27.759227-89.797166-27.759227-137.461745 0-34.233686 5.108345-68.423369 15.181771-101.617375 0.191358-0.630357 0.415462-1.25048 0.672312-1.857301 6.261611-14.797008 16.156982-19.971867 23.355941-21.707394 15.544021-3.752464 28.114314 5.92699 30.476105 7.90504 0.421602 0.353041 0.827855 0.724501 1.215688 1.113357l156.502403 156.506497c42.113143 42.11212 110.63475 42.111097 152.74687 0l14.932084-14.932084c42.113143-42.113143 42.113143-110.636796 0-152.74994L275.221272 110.122073c-2.995217-2.952238-16.23066-17.145495-12.616342-33.943067 1.248434-5.803169 5.612834-16.364713 20.808932-21.902846 0.297782-0.107447 0.596588-0.208754 0.899486-0.301875 33.685193-10.372232 68.334341-15.631002 102.981443-15.631002 47.664579 0 93.913926 9.338692 137.462768 27.758204 42.053792 17.787108 79.81686 43.24799 112.243387 75.673493 32.425504 32.426527 57.884338 70.190619 75.671447 112.24441 18.419512 43.548842 27.759227 89.798189 27.759227 137.463792 0 47.240931-9.277294 93.215008-27.574009 136.647193-0.413416 0.980327-0.909719 1.923816-1.483794 2.820232-3.445472 5.384637-3.813862 9.953699-1.227967 15.277962 2.122337 4.373611 5.500271 7.500834 5.531993 7.526417 0.430812 0.360204 0.845251 0.738827 1.242294 1.13587l205.101261 205.101261c24.787546 24.781406 36.160571 60.248176 30.423917 94.872764-1.687432 10.187013-11.313673 17.072841-21.50171 15.389502-10.187013-1.687432-17.076934-11.314697-15.389502-21.50171 3.76986-22.74912-3.696182-46.046732-19.97289-62.318324L690.99365 581.846788c-12.778024-11.129478-30.560016-38.879496-12.039197-69.567425 15.981996-38.420031 24.083511-79.051427 24.083511-120.79618 0-84.339874-32.844036-163.630754-92.479251-223.268016C550.922475 108.578927 471.632617 75.735915 387.29479 75.735915c-27.644617 0-55.31277 3.759627-82.392522 11.18576l150.780075 150.780075c27.463492 27.463492 42.587957 63.978127 42.587957 102.81669s-15.124466 75.352175-42.587957 102.81669l-14.932084 14.932084c-56.69321 56.692187-148.936077 56.691163-205.629287 0L83.412758 306.553884c-7.872295 27.833929-11.861142 56.373938-11.861142 84.929298 0 84.337827 32.843012 163.626661 92.480274 223.263923 59.636239 59.635215 138.926096 92.479251 223.263923 92.479251 46.034453 0 90.434685-9.717316 131.97887-28.884864 24.868387-12.609179 46.728254 0.959861 55.353702 10.410094l209.227232 209.41859c15.799848 15.799848 37.440727 23.179932 59.57177 20.366863 9.735735-3.145643 57.338916-17.845437 106.856703-21.383006 10.30367-0.728594 19.246343 7.017834 19.981077 17.317411 0.735757 10.299577-7.017834 19.246343-17.317411 19.981077-49.66105 3.545756-98.847286 19.930935-99.338473 20.09671-1.049912 0.353041-2.130524 0.61296-3.226484 0.773619C845.128123 956.099541 839.85298 956.481234 834.603419 956.481234z")
        
        svgs.style("cursor", "pointer")
        svgs.on("click", (ev, d) => {
            if(d.id === that.current_class) return;
            if(d.id === -2){
                // set background frame
                let history = that.parent.history;
                if(history.length === 0 || !history[history.length - 1].is_comparison) return;
                that.parent.remove_action();
            }
            else{
                // TODO: set other categroy
                return;
            }
            // if(that.parent.in_length_compare && that.parent.compare_step === d.step){
            //     that.parent.set_in_length_compare(false);
            //     that.parent.set_compare_step(-1);
            //     that.sub_component_update();
            // }
            // else{
            //     console.log("click compare", ev, d)
            //     that.parent.set_compare_step(d.step);
            //     that.sub_component_update();
            //     that.parent.fetch_step_labels(d.step);
            // }
        })
    };

    that.update = function () {
        that.labels_group
            .selectAll("g.work-labels")
            .transition()
            .duration(that.update_ani)
            .attr("fill", (d) => d.id == that.current_class ? Global.DarkOrange: "black");
    };

    that.remove = function () {
        that.e_lines
            .exit()
            .transition()
            .duration(this.remove_ani)
            .style("opacity", 0)
            .remove();
    };
}

export default LabelsRender;