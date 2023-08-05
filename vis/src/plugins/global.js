import * as d3 from "d3";

const GrayColor = "#7f7f7f";
// const DarkGray = "rgb(211, 211, 229)";
const DeepGray = "rgb(50, 50, 50)";
const LightGray = "#EBEBF3";
const Orange = "#ffa953";
const DarkOrange = "#ff7f0e";
//["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
const DarkGreen = "rgb(112, 173, 71)";
const Red = "rgb(237, 41, 57)";
const BoxRed = "#93ff2f";
const LightBlue = "rgb(227, 234, 244)";
const Blue = "rgb(1, 103, 243)";
const BlueGray = d3.interpolate(LightGray, Blue)(0.3);
const DarkBlueGray = d3.interpolate(LightGray, Blue)(0.6);
const Blue40 = d3.interpolate('white', Blue)(0.4);
const Blue60 = d3.interpolate('white', Blue)(0.6);
const MidBlue = d3.interpolate('white', Blue)(0.8);
const Purple = "#7376d0";
const Animation = 1000;
const QuickAnimation = 500;
const WindowHeight = window.innerHeight - 10;
const linked_highlight = true;
const compare_icon = '<svg t="1643350929500" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2026" xmlns:xlink="http://www.w3.org/1999/xlink" width="22" height="22"><path d="M528 128a48 48 0 0 1 47.776 43.392L576 176V256h272a48 48 0 0 1 47.776 43.392L896 304v544a48 48 0 0 1-43.392 47.776L848 896h-352a48 48 0 0 1-47.776-43.392L448 848V768H176a48 48 0 0 1-47.776-43.392L128 720v-544a48 48 0 0 1 43.392-47.776L176 128h352z m304 192h-256v400a48 48 0 0 1-43.392 47.776L528 768H512v64h320v-224h-114.752l35.904 35.872a32 32 0 0 1-42.24 47.936l-3.04-2.656-90.496-90.528a32 32 0 0 1-2.656-42.24l2.656-3.008 90.496-90.528a32 32 0 0 1 47.936 42.24l-2.656 3.04-35.904 35.84L832 544v-224zM512 192H192v224h173.248l-35.872-35.872a32 32 0 0 1 45.248-45.28l90.528 90.528a32 32 0 0 1 0 45.248l-90.528 90.528a32 32 0 0 1-45.248-45.28L365.248 480H192v224h320V192z" p-id="2027" style="fill:rgb(120, 120, 120)"></path></svg>'
const compare_cursor = 'data:image/svg+xml;base64,'  + window.btoa(unescape(encodeURIComponent(compare_icon)));
const logic_height = 1080
const logic_width = 1920

const deepCopy = function (obj) {
    let _obj = Array.isArray(obj) ? [] : {}
    for (let i in obj) {
        _obj[i] = typeof obj[i] === 'object' ? deepCopy(obj[i]) : obj[i]
    }
    return _obj
};

function pos2str_inverse(pos) {
    return pos.y + "," + pos.x;
}

function pos2str(pos) {
    return pos.x + "," + pos.y;
}

const collapse_icon = function(x, y, type, basic_ratio){
    basic_ratio = basic_ratio || 4;
    if (type === 0) { // collapsed
        let ratio = basic_ratio * 1;
        let p1 = {x: x + ratio, y: y};
        let p2 = {x: x - 0.5 * ratio, y: y - 1.866 * ratio};
        let p3 = {x: x - 0.5 * ratio, y: y + 1.866 * ratio};
        return "M" + pos2str(p2) + "L" + pos2str(p1) + "L" + pos2str(p3);
    }
    else if (type === 1){
        let ratio = basic_ratio * 1;
        let p1 = {x: x, y: y + ratio};
        let p2 = {x: x + 1.866 * ratio, y: y - 0.5 * ratio};
        let p3 = {x: x - 1.866 * ratio, y: y - 0.5 * ratio};
        return "M" + pos2str(p2) + "L" + pos2str(p1) + "L" + pos2str(p3);
    }
    else{
        return 1;
    }
}

const node_icon = function(x, y, type, basic_ratio){
    basic_ratio = basic_ratio || 3;
    if (type === 0){ // collapsed
        let ratio = basic_ratio * 1.8;
        let p1 = {x: x + ratio, y: y};
        let p2 = {x: x - 0.5 * ratio, y: y - 0.866 * ratio};
        let p3 = {x: x - 0.5 * ratio, y: y + 0.866 * ratio};
        return "M" + pos2str(p2) + "L" + pos2str(p1) + "L" + pos2str(p3);
    }
    else if (type === 1){ //expanded
        let ratio = basic_ratio * 1.8;
        let p1 = {x: x, y: y + ratio};
        let p2 = {x: x + 0.866 * ratio, y: y - 0.5 * ratio};
        let p3 = {x: x - 0.866 * ratio, y: y - 0.5 * ratio};
        return "M" + pos2str(p2) + "L" + pos2str(p1) + "L" + pos2str(p3);
    }
    else if (type === 2){ // leaf node
        let ratio = 0; // hidden
        // let ratio = basic_ratio;
        return "M " + (x - ratio) + ", " + (y) + 
            "a" + ratio + ", " + ratio + " 0 1, 0 " + (ratio * 2) + ", 0" +  
            "a" + ratio + ", " + ratio + " 0 1, 0 " + (- ratio * 2) + ", 0"; 
    }
    else if (type === -1){
        return "M 0,0 L 0,0";
    }
    else{
        return 1;
    }
}

function plus_path_d(start_x, start_y, width, height, k) {
    let sum_k = 2 * k + 1;
    let x = [start_x, start_x + k / sum_k * width, start_x + (k + 1) / sum_k * width, start_x + width];
    let y = [start_y, start_y + k / sum_k * height, start_y + (k + 1) / sum_k * height, start_y + height];
    let d = `M${x[0]},${y[1]}`;
    d += `L${x[1]},${y[1]}`;
    d += `L${x[1]},${y[0]}`;
    d += `L${x[2]},${y[0]}`;
    d += `L${x[2]},${y[1]}`;
    d += `L${x[3]},${y[1]}`;
    d += `L${x[3]},${y[2]}`;
    d += `L${x[2]},${y[2]}`;
    d += `L${x[2]},${y[3]}`;
    d += `L${x[1]},${y[3]}`;
    d += `L${x[1]},${y[2]}`;
    d += `L${x[0]},${y[2]}`;
    d += `L${x[0]},${y[1]}`;
    return d;
}

function minus_path_d(start_x, start_y, width, height, k){
    let sum_k = 2 * k + 1;
    let x = [start_x, start_x  + width];
    let y = [start_y + k / sum_k * height, start_y + (k + 1) / sum_k * height];
    let d = `M${x[0]},${y[0]}`; 
    d += `L${x[1]},${y[0]}`
    d += `L${x[1]},${y[1]}`
    d += `L${x[0]},${y[1]}`;
    return d;
}

const getTextWidth = function (text, font) {
    let canvas = getTextWidth.canvas || (getTextWidth.canvas = document.createElement("canvas"));
    let context = canvas.getContext("2d");
    context.font = font;
    return context.measureText(text).width;
}


function disable_global_interaction() {
    d3.select(".loading")
        .style("display", "block")
        .style("opacity", 0.5);
}

function enable_global_interaction(delay) {
    delay = delay || 1;
    d3.select(".loading")
        .transition()
        .duration(1)
        .delay(delay)
        .style("display", "none")
        .style("opacity", 1);

}

function begin_loading() {
    // $(".loading").show();
    // $(".loading-svg").show();
    d3.select(".loading")
        .style("display", "block");
    d3.select(".loading-svg")
        .style("display", "block");
}

function end_loading(delay) {
    delay = delay || 1;
    console.log("delay", delay);
    d3.select(".loading")
        .transition()
        .duration(1)
        .delay(delay)
        .style("display", "none");
    d3.select(".loading-svg")
        .transition()
        .duration(1)
        .delay(delay)
        .style("display", "none");
}

const path_curve = function (points) {
    // const movePoint = (p, x, y, s) => {
    //     return { x: p.x * s + x, y: p.y * s + y }
    // };
    // points = points.map(p => movePoint(p, transX, transY, scale))

    let len = points.length;
    if (len === 0) { return "" }
    let start = `M ${points[0].x} ${points[0].y}`,
        vias = [];

    const getInter = (p1, p2, n) => {
        return `${p1.x * n + p2.x * (1 - n)} ${p1.y * n + p2.y * (1 - n)}`
    };

    const getCurve = (points) => {
        let vias = [],
            len = points.length;
        const ratio = 0.5;
        for (let i = 0; i < len - 2; i++) {
            let p1, p2, p3, p4, p5;
            if (i === 0) {
                p1 = `${points[i].x} ${points[i].y}`
            } else {
                p1 = getInter(points[i], points[i + 1], ratio)
            }
            p2 = getInter(points[i], points[i + 1], 1 - ratio);
            p3 = `${points[i + 1].x} ${points[i + 1].y}`;
            p4 = getInter(points[i + 1], points[i + 2], ratio);
            if (i === len - 3) {
                p5 = `${points[i + 2].x} ${points[i + 2].y}`
            } else {
                p5 = getInter(points[i + 1], points[i + 2], 1 - ratio)
            }
            let cPath = `M ${p1} L${p2} Q${p3} ${p4} L${p5}`;
            vias.push(cPath);
        }
        return vias
    };
    vias = getCurve(points);
    let pathData = `${start}  ${vias.join(' ')}`;
    return pathData;
};

export {
    GrayColor,
    DeepGray,
    LightGray,
    Orange,
    DarkOrange,
    DarkGreen,
    BlueGray,
    DarkBlueGray,
    BoxRed,
    Animation,
    QuickAnimation,
    WindowHeight,
    Red,
    Blue,
    Blue40,
    Blue60,
    MidBlue,
    Purple,
    LightBlue,
    linked_highlight,
    compare_cursor,
    collapse_icon,
    plus_path_d,
    minus_path_d,
    node_icon,
    path_curve,
    deepCopy,
    getTextWidth,
    begin_loading,
    end_loading,
    disable_global_interaction,
    enable_global_interaction,
    logic_height,
    logic_width,
}