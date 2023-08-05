import * as d3 from "d3";
import { svg } from "d3";
import * as Global from "./global";

const HistoryRender = function (parent) {
    let that = this;
    that.parent = parent;
    that.line_height = 40;
    that.line_margin = 10;
    that.width_margin = 20;
    that.width_svg = 24;
    that.width_ratios = [0.25, 0.3, 0.3, 0.1];
    that.cur_height = that.line_height + 2 * that.line_margin;
    that.font_size = 18;


    that.update_info_from_parent = function () {
        that.server_url = that.parent.server_url;
        that.history_group = that.parent.history_group;
        that.history_data = that.parent.work_history;

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
        console.log(that.history_data)
        let data = that.layout()

        // update view
        that.e_lines = that.history_group
            .selectAll(".work-history")
            .data(data);

        that.create();
        that.update();
        that.remove();

        return that.cur_height;
    }

    that.layout = function(){
        let data = [];
        for(let i = 0; i < that.history_data.boundary.length; i++){
            data.push({
                step: i,
                boundary: that.history_data.boundary[i],
                alignment: that.history_data.must_link[i] + that.history_data.must_not_link[i]
            })
        }

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
            step: -1,
            x: 0,
            y: that.line_margin * 2 / 3,
        })
        data.forEach((d, index) => {
            d.type = 'data',
            d.x = 0,
            d.y = that.line_margin * 2 / 3 + (that.line_height + that.line_margin) * (index + 1)
            ret.push(d);
        })
        if(ret.length > 1) ret[ret.length - 1].type = 'final';
        that.cur_height = (that.line_height + that.line_margin) * ret.length;
        return ret;
    }

    that.create = function () {
        console.log("history_create")
        let line_groups = that.e_lines
            .enter()
            .append("g")
            .attr("class", "work-history")
            .attr("id", (d) => "work-history" + d.step)
            .attr("transform", (d) => "translate(" + d.x + ", " + d.y + ")")
            .attr("fill", d => that.parent.in_length_compare && d.step === that.parent.compare_step ? Global.DarkOrange: "black");

        line_groups.append("text")
            .text(d => d.type === 'schema' ? 'Id' : `${d.step}`)
            .attr("text-anchor", "middle")
            .attr("font-size", that.font_size)
            .attr("font-weight", d => d.type === "schema" ? "bold" : "normal")
            .attr("opacity", 0.6)
            .attr("transform", `translate(${that.widths[0]}, ${(that.line_height - that.font_size) / 2})`);

        line_groups.append("text")
            .text(d => d.type === 'schema' ? 'Boundary' : d.boundary)
            .attr("text-anchor", "middle")
            .attr("font-size", that.font_size)
            .attr("font-weight", d => d.type === "schema" ? "bold" : "normal")
            .attr("opacity", 0.6)
            .attr("transform", `translate(${that.widths[1]}, ${(that.line_height - that.font_size) / 2})`);

        line_groups.append("text")
            .text(d => d.type === 'schema' ? 'Alignment' : d.alignment)
            .attr("text-anchor", "middle")
            .attr("font-size", that.font_size)
            .attr("font-weight", d => d.type === "schema" ? "bold" : "normal")
            .attr("opacity", 0.6)
            .attr("transform", `translate(${that.widths[2]}, ${(that.line_height - that.font_size) / 2})`);

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
            .attr("transform", `translate(${that.widths[3] - that.width_svg / 2}, ${(that.line_height - that.width_svg) / 2 - that.width_svg + 7})`)
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
            .attr("d", "M99.57888 147.39456a4.77696 4.77696 0 0 1 4.76672-4.79744h30.39744a4.78208 4.78208 0 0 1 4.79744 4.76672v741.29408h807.8848a4.79232 4.79232 0 0 1 4.79232 4.79232v30.37696a4.80256 4.80256 0 0 1-4.79232 4.79232H99.57888V147.39456z")
        svgs.append("path")
            .attr("d", "M192.34816 303.24224c0-3.03616 1.15712-6.07744 3.47648-8.39168a11.81184 11.81184 0 0 1 8.39168-3.47648H312.2176c3.03616 0 6.07232 1.15712 8.38656 3.47648 2.31936 2.31936 3.47648 5.3504 3.47648 8.39168V862.208H192.34816V303.24224z m393.99936 164.6592c0-3.03616 1.16224-6.07744 3.4816-8.39168a11.8016 11.8016 0 0 1 8.39168-3.47648h108.00128a11.8784 11.8784 0 0 1 11.86816 11.86816v394.29632h-131.73248V467.90144h-0.01024zM353.29536 512v350.19776h131.73248V512H353.29536z")
        svgs.append("path")
            .attr("d", "M462.68416 841.088h-87.04v-306.90304h87.04v306.90304z")
            .attr("fill", "#FFFFFF")
        svgs.append("path")
            .attr("d", "M737.29536 659.11808V863.232h131.73248v-204.1088h-131.73248z")
        svgs.append("path")
            .attr("d", "M846.67904 845.97248h-87.04v-168.96h87.04v168.96z")
            .attr("fill", "#FFFFFF")
        
        svgs.style("cursor", "pointer")
        svgs.on("click", (ev, d) => {
            if(that.parent.in_length_compare && that.parent.compare_step === d.step){
                that.parent.set_in_length_compare(false);
                that.parent.set_compare_step(-1);
                that.sub_component_update();
            }
            else{
                console.log("click compare", ev, d)
                that.parent.set_compare_step(d.step);
                that.sub_component_update();
                that.parent.fetch_step_history(d.step);
            }
        })
    };

    that.update = function () {
        that.history_group
            .selectAll("g.work-history")
            .transition()
            .duration(that.update_ani)
            .attr("fill", d => d.step > -1 && d.step === that.parent.compare_step ? Global.DarkOrange: "black");
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

export default HistoryRender;