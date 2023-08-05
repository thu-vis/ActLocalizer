const ClassLayout = function(parent){
    let that = this;
    that.parent = parent;

    that.node_width = 49;
    that.node_height = 70;
    that.node_h_margin = 15;
    that.node_v_margin = 20;
    
    that.update_info_from_parent = function(){
        // animation
        that.create_ani = that.parent.create_ani;
        that.update_ani = that.parent.update_ani;
        that.remove_ani = that.parent.remove_ani;
        that.layout_width = that.parent.layout_width;
        that.layout_height = that.parent.layout_height;
    }

    this.update_layout = function(class_info){
        that.update_info_from_parent();
        
        that.num_per_row = (that.layout_width + that.node_h_margin) /
            (that.node_width + that.node_h_margin);
        that.num_per_row = parseInt(that.num_per_row);
        that.left_margin = that.layout_width - that.num_per_row * that.node_width
            - (that.num_per_row - 1) * that.node_h_margin;
        that.left_margin /= 2;

        let max_num = Math.max(...class_info.map(d => d.num));

        class_info.sort((a, b) => b.entropy - a.entropy);
        
        class_info.forEach((d, i) => {
            let row_idx = parseInt(i / that.num_per_row);
            let col_idx = i - row_idx * that.num_per_row;
            d.x = that.left_margin + col_idx * (that.node_width + that.node_h_margin);
            d.y = row_idx * (that.node_height + that.node_v_margin);
            // d.width = that.node_width * that.scale_function(d.num / max_num);
            d.width = that.node_width;
            d.height = that.node_height;
            d.img_width = d.width * 0.9;
            d.img_height = d.img_width;
            d.img_x = d.width * 0.1 / 2;
            d.img_y = d.width * 0.1 / 2;
            d.text_x = d.width / 2;
            d.text_y = d.height;
        });

        let max_height = Math.max(...class_info.map(d => d.height))
             + that.node_v_margin;
        return [class_info, max_height];
    };

    that.scale_function = function(x){
        return Math.pow(x, 0.9);
    };

}

export {ClassLayout};