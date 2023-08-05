import * as d3 from "d3";
import * as Global from "./global";
import { ClassLayout } from "./layout_class";

const ClassRender = function (parent) {
    let that = this;
    that.parent = parent;
    that.layout = new ClassLayout(that);

    that.update_info_from_parent = function () {
        that.server_url = that.parent.server_url;
        that.class_group = that.parent.class_group;

        that.class_info = that.parent.class_info;
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
        let data = Global.deepCopy(that.class_info);
        [data, that.max_height] = that.layout.update_layout(data);

        // update view
        that.e_classes = that.class_group
            .selectAll(".action-class")
            .data(data, d => d.name);

        that.create();
        that.update();
        that.remove();

        return that.max_height;
    }

    that.create = function () {
        let node_groups = that.e_classes
            .enter()
            .append("g")
            .attr("class", "action-class")
            .attr("id", (d) => "class-id-" + d.id)
            .attr("transform", (d) => "translate(" + d.x + ", " + d.y + ")");
        node_groups
            .style("opacity", 0)
            .on("mouseover", function(ev) {
              that.highlight(ev);
            })
            .on("mouseout", function() {
              that.dehighlight();
            })
            .on("click", function(ev, d){
                that.parent.set_selected_class(d.id);
            })
            .transition()
            .duration(that.create_ani)
            .delay(that.remove_ani + that.update_ani)
            .style("opacity", 1);
        node_groups
            .append("title")
            .style("font-size", "0.835vw")
            .text((d) => d.full_name);
        node_groups
            .append("rect")
            .attr("class", "background")
            // .attr("rx", d => d.width / 15)
            // .attr("ry", d => d.width / 15)
            .attr("rx", 0)
            .attr("ry", 0)
            .attr("x", d => 0)
            .attr("y", d => 0)
            .attr("height", d => d.height)
            .attr("width", d => d.width)
            .style("fill", "white")
            .style("stroke-width",  d => d.id === that.parent.selected_class ? 3 : 1)
            .style("stroke", d => d.id === that.parent.selected_class ? Global.Orange : Global.GrayColor);
        

        node_groups
            .append("image")
            .attr("class", "action-icon")
            .attr("href", d => that.server_url + "/Corpus/ActionIcon?filename=" + d.full_name)
            .attr("height", d => d.img_height)
            .attr("width", d => d.img_width)
            .attr("x", d => d.img_x)
            .attr("y", d => d.img_y)

        node_groups
            .append("text")
            .attr("class", "action-num")
            .text((d) => {
                return "# " + d.num;
            })
            .attr("text-anchor", "middle")
            .attr("x", d => d.text_x)
            .attr("dy", d => d.height - 8)
            .style("fill", Global.DeepGray);


        node_groups
            .append("rect")
            .attr("class", "entropy")
            .attr("x", 0)
            .attr("y", d => (1 - d.entropy) * d.height)
            .attr("width", d => d.width)
            .attr("height", d => d.entropy * d.height)
            .style("stroke", 0)
            .style("fill", Global.Blue40)
            .style("opacity", 0.3);
    };

    that.update = function () {
        that.e_classes
            .transition()
            .duration(that.update_ani)
            .delay(that.remove_ani)
            .attr("transform", (d) => "translate(" + d.x + ", " + d.y + ")");
        
        that.e_classes
            .select("rect.background")
            .style("stroke-width",  d => d.id === that.parent.selected_class ? 3 : 1)
            .style("stroke", d => d.id === that.parent.selected_class ? Global.Orange : Global.GrayColor);

    };

    that.remove = function () {
        that.e_classes
            .exit()
            .transition()
            .duration(this.remove_ani)
            .style("opacity", 0)
            .remove();
    };

    that.highlight = function(ev){
        let element = ev.target;
        while (element.tagName !== "g" || element.className.baseVal !== "action-class") {
          element = element.parentElement;
        }
        let self = d3.select(element);
        self.selectAll("rect.background")
            .style("stroke", Global.Orange)
            .style("stroke-width", 3);

    };

    that.dehighlight = function(){
        that.class_group
            .selectAll("rect.background")
            .style("stroke", Global.GrayColor)
            .style("stroke-width", 1);
        that.class_group
            .select("#class-id-" + that.parent.selected_class)
            .select("rect.background")
            .style("stroke", Global.Orange)
            .style("stroke-width", 3);
    };
}

export default ClassRender;