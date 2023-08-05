
function limitPrecision(num, precision = 2) {
    return Math.round(num * 10 ** precision) / 10 ** precision;
}

function statisticalDivision(lengths, divide_target = 6) {
    let max_length = Math.max(...lengths);
    let min_length = Math.min(...lengths);
    let max_delta = max_length - min_length + 1;
    console.assert(max_length > 0, "max length must > 0");
    // let start = min_length - 1, delta = Math.floor(max_delta / divide_target);
    // if (delta === 0) {
    //     divide_target = max_delta;
    //     delta = 1;
    // }
    let delta = 1;
    let bars = [];
    for (let i = 0; i < divide_target; i++) {
        let bar = {};
        // bar.left = start + 1;
        // bar.right = (i === divide_target - 1) ? max_length : start + delta;
        // bar.count = 0;
        // start = bar.right;
        bar.left = i * delta + 1;
        bar.right = i * delta + 2;
        bar.count = 0;
        if (i === divide_target - 1) bar.right = 100;
        bars.push(bar);
    }

    lengths.forEach(d => {
        bars[Math.min(divide_target - 1, d - 1)].count += 1;
    })
    bars[divide_target - 1].left += "+";
    return bars;
}

function scaleClustersBars(clusters, compare=false) {
    let max_count = 0;
    clusters.forEach(d => {
        d.bars.forEach((e, index) => {
            if (e.count > max_count)
                max_count = e.count;
            if(compare){
                e.ccount = d.cbars[index].count;
                if (e.ccount > max_count)
                    max_count = e.ccount;
            }
        })
    })
    max_count = Math.max(max_count, 10);

    const gap = 4, width = 15, height = 60;
    const alpha = height / max_count;

    clusters.forEach(d => {
        d.bar_height = height;
        d.cur_height = 0;
        d.bar_width = d.bars.length * (gap + width) - gap;
        d.bars.forEach((e, index) => {
            e.x = index * gap + index * width;
            e.height = e.count * alpha;
            e.width = width;
            e.y = height - e.height;
            if(e.height > d.cur_height)
                d.cur_height = e.height;
            if(compare){
                e.cx = index * gap + index * width + width / 2;
                e.width = (width - gap) / 2;
                e.cheight = e.ccount * alpha;
                e.cy = height - e.cheight;
            }
        })
    })
}

function rotate(point, center, radians) {
    const cos = Math.cos(radians),
        sin = Math.sin(radians),
        nx = cos * (point.x - center.x) + sin * (point.y - center.y) + center.x,
        ny = cos * (point.y - center.y) - sin * (point.x - center.x) + center.y;
    return { x: nx, y: ny };
}

// xs and lines should be sorted array.
function lowerIntersect(xs, lines) {
    const min_value = 1e10
    let left = 0, right = 0
    let ret = []
    for (let x of xs) {
        while (left < lines.length && lines[left][1][0] < x) ++left
        while (right < lines.length && lines[right][0][0] <= x) ++right
        let min_y = min_value
        for (let i = left; i < right; ++i) {
            if (x >= lines[i][0][0] && x <= lines[i][1][0]) {
                let y = Math.min(lines[i][0][1], lines[i][1][1])
                if (lines[i][0][0] != lines[i][1][0]) {
                    y = (x - lines[i][0][0]) / (lines[i][1][0] - lines[i][0][0])
                        * (lines[i][1][1] - lines[i][0][1]) + lines[i][0][1]
                }
                if (y < min_y) {
                    min_y = y
                }
            }
        }
        ret.push(min_y)
    }
    if (Math.min(...ret) == min_value) {
        const new_min_value = Math.min(...lines.map(d => Math.min(d[0][1], d[1][1])))
        ret = ret.map(_ => new_min_value)
    }
    return ret
}

// xs and lines should be sorted array.
function upperIntersect(xs, lines) {
    let left = 0, right = 0
    const max_value = -1e10
    let ret = []
    for (let x of xs) {
        while (left < lines.length && lines[left][1][0] < x) ++left
        while (right < lines.length && lines[right][0][0] <= x) ++right
        let max_y = max_value
        for (let i = left; i < right; ++i) {
            if (x >= lines[i][0][0] && x <= lines[i][1][0]) {
                let y = Math.max(lines[i][0][1], lines[i][1][1])
                if (lines[i][0][0] != lines[i][1][0]) {
                    y = (x - lines[i][0][0]) / (lines[i][1][0] - lines[i][0][0])
                        * (lines[i][1][1] - lines[i][0][1]) + lines[i][0][1]
                }
                if (y > max_y) {
                    max_y = y
                }
            }
        }
        ret.push(max_y)
    }
    if (Math.max(...ret) == max_value) {
        const new_max_value = Math.max(...lines.map(d => Math.max(d[0][1], d[1][1])))
        ret = ret.map(_ => new_max_value)
    }
    return ret
}

function areaMargin(xs, lower_area, upper_area) {
    let lower_y = upperIntersect(xs, lower_area)
    let upper_y = lowerIntersect(xs, upper_area)
    return upper_y.map((d, index) => d - lower_y[index])
}

function placeImages(points, lower_area, upper_area, { grid_size = 4, area_width = 1000, image_width = 80, image_height = 80, padding = 4 }) {
    let grid_xs = []
    let min_gap = Math.floor(image_width / grid_size)
    let min_padding = Math.ceil(padding / grid_size)
    for (let x = -image_width / 2; x < area_width; x += grid_size) {
        grid_xs.push(x)
    }
    let lower_y = lower_area ? upperIntersect(grid_xs, lower_area) : grid_xs.map(_ => -15)
    let upper_y = upper_area ? lowerIntersect(grid_xs, upper_area) : grid_xs.map(_ => 0)
    let grid_can_place = grid_xs.map((d, i) => (upper_y[i] - lower_y[i] >= image_height) ? 1 : 0)
    for (let i = 1; i < grid_can_place.length; ++i) {
        if (grid_can_place[i]) {
            grid_can_place[i] += grid_can_place[i - 1]
        }
    }
    grid_can_place = grid_can_place.map(d => d > min_gap)
    let f = [], pre = []
    for (let i = 0; i < points.length; ++i) {
        f.push(new Float32Array(grid_xs.length))
        pre.push(new Int32Array(grid_xs.length))
        for (let j = 0; j < grid_xs.length; ++j) {
            f[i][j] = 1e10;
        }
    }

    let l = min_gap + min_padding + 2
    let n = points.length
    // console.log(points.map(d => d.x))
    for (let i = 0; i < n; ++i) {
        let min_f = 1e10, min_pre = -1
        if (i == 0) {
            min_f = 0
        }
        for (let j = 0, k = 0; j < grid_xs.length; ++j) {
            if (i > 0) {
                while (k + l < j) {
                    if (f[i - 1][k] < min_f) {
                        min_f = f[i - 1][k]
                        min_pre = k
                    }
                    ++k
                }
            }
            if (!grid_can_place[j]) {
                continue
            }
            let x = grid_xs[j]
            let cost = Math.abs(x - points[i].x) ** 2
            f[i][j] = cost + min_f
            pre[i][j] = min_pre
        }
    }

    let min_f = 1e10, k = -1
    for (let j = 0; j < grid_xs.length; ++j) {
        if (f[n - 1][j] < min_f) {
            min_f = f[n - 1][j]
            k = j
        }
    }
    if (min_f == 1e10) {
        return null
    }
    let ret = []
    let dk = Math.floor(image_width / grid_size / 2)
    for (let i = n - 1; i >= 0; --i) {
        let x = grid_xs[k]
        let y1 = upper_y[k], y2 = lower_y[k]
        for (let j = Math.max(0, k - dk); j <= Math.min(grid_xs.length - 1, k + dk); ++j) {
            y1 = Math.min(y1, upper_y[j])
            y2 = Math.max(y2, lower_y[j])
        }
        let y = Math.max(y1 - 10, y2 + image_height + 5)
        ret.push({ x, y })
        k = pre[i][k]
    }
    ret = ret.reverse()
    return ret.map((d, i) => ({
        x: points[i].x - d.x,
        y: Math.max(5, points[i].y - d.y),
    }))
}

function generateSortedEdges(pointset) {
    let lines = []
    for (let i = 0; i < pointset.length; ++i) {
        let p1 = pointset[i]
        let p2 = pointset[i == (pointset.length - 1) ? 0 : (i + 1)]
        if (p1[0] > p2[0]) { let t = p1; p1 = p2; p2 = t; }
        lines.push([p1, p2])
    }
    return lines.sort((a, b) => a[0][0] != b[0][0] ? (a[0][0] - b[0][0]) : (a[1][0] - b[1][0]))
}

function vec(p1, p2) {
    // Returns a ( p1 o-> p2 ) vector object
    // given p1 and p2 objects with the shape {x: number, y: number}

    // horizontal and vertical vector components
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;

    // magnitude of the vector (distance)
    const mag = Math.hypot(dx, dy);

    // unit vector
    const unit = mag !== 0 ? { x: dx / mag, y: dy / mag } : { x: 0, y: 0 };

    // normal vector
    const normal = rotate(unit, { x: 0, y: 0 }, Math.PI / 2);

    // Angle in radians
    const radians = Math.atan2(dy, dx);
    // Normalize to clock-wise circle
    const normalized = 2 * Math.PI + (Math.round(radians) % (2 * Math.PI));
    // Angle in degrees
    const degrees = (180 * radians) / Math.PI;
    const degreesNormalized = (360 + Math.round(degrees)) % 360;
    // const degrees2 = ( 360 + Math.round(degrees) ) % 360;

    return {
        dx,
        dy,
        mag,
        unit,
        normal,
        p1: { ...p1 },
        p2: { ...p2 },
        angle: {
            radians,
            normalized,
            degrees,
            degreesNormalized,
        },
    };
}
function roundSvgPath(pathData, radius, style) {
    if (!pathData || radius === 0) {
        return pathData;
    }

    // Get path coordinates as array of {x, y} objects
    let pathCoords = pathData
        .split(/[a-zA-Z]/)
        .reduce(function (parts, part) {
            var match = part.match(/(-?[\d]+(\.\d+)?)/g);
            if (match) {
                // Cast to number before pushing coordinate
                parts.push({
                    x: +match[0],
                    y: +match[1],
                });
            }
            return parts;
        }, [])
        .filter((e, i, arr) => {
            // Remove consecutive duplicates
            const prev = arr[i === 0 ? arr.length - 1 : i - 1];
            return e.x !== prev.x || e.y !== prev.y;
        });

    // Build rounded path
    const path = [];
    for (let i = 0; i < pathCoords.length; i++) {
        // Get current point and the next two (start, corner, end)
        const c2Index = (i + 1) % pathCoords.length;
        const c3Index = (i + 2) % pathCoords.length;

        const c1 = pathCoords[i];
        const c2 = pathCoords[c2Index];
        const c3 = pathCoords[c3Index];

        // Vectors from middle point (c2) to each ends
        const vC1c2 = vec(c2, c1);
        const vC3c2 = vec(c2, c3);

        const angle = Math.abs(
            Math.atan2(
                vC1c2.dx * vC3c2.dy - vC1c2.dy * vC3c2.dx, // cross product
                vC1c2.dx * vC3c2.dx + vC1c2.dy * vC3c2.dy // dot product
            )
        );

        // Limit radius to 1/2 the shortest edge length to:
        // 1. allow rounding the next corner as much as the
        //    one we're dealing with right now (the 1/2 part)
        // 2. draw part of a circle and not an ellipse, hence
        //    we keep the shortest edge of the two
        const cornerLength = Math.min(radius, vC1c2.mag / 2, vC3c2.mag / 2);

        // Find out tangential circle radius
        const bc = cornerLength;
        const bd = Math.cos(angle / 2) * bc;
        const fd = Math.sin(angle / 2) * bd;
        const bf = Math.cos(angle / 2) * bd; // simplify from abs(cos(PI - angle / 2))
        const ce = fd / (bf / bc);
        const be = bd / (bf / bc);
        const a = ce;

        // Compute control point distance to create a circle
        // with quadratic bezier curves
        const numberOfPointsInCircle = (2 * Math.PI) / (Math.PI - angle);
        let idealControlPointDistance;

        if (style === "circle") {
            // Strictly geometric
            idealControlPointDistance =
                (4 / 3) * Math.tan(Math.PI / (2 * numberOfPointsInCircle)) * a;
        } else if (style === "approx") {
            // Serendipity #1 rounds the shape more naturally
            idealControlPointDistance =
                (4 / 3) *
                Math.tan(Math.PI / (2 * ((2 * Math.PI) / angle))) *
                cornerLength *
                (angle < Math.PI / 2 ? 1 + Math.cos(angle) : 2 - Math.sin(angle));
        } else if (style === "hand") {
            // Serendipity #2 'hands free' style
            idealControlPointDistance =
                (4 / 3) *
                Math.tan(Math.PI / (2 * ((2 * Math.PI) / angle))) *
                cornerLength *
                (2 + Math.sin(angle));
        }

        // First point and control point
        const cpDistance = cornerLength - idealControlPointDistance;

        // Start of the curve
        let c1c2curvePoint = {
            x: c2.x + vC1c2.unit.x * cornerLength,
            y: c2.y + vC1c2.unit.y * cornerLength,
        };
        // First control point
        let c1c2curveCP = {
            x: c2.x + vC1c2.unit.x * cpDistance,
            y: c2.y + vC1c2.unit.y * cpDistance,
        };

        // Second point and control point
        // End of the curve
        let c3c2curvePoint = {
            x: c2.x + vC3c2.unit.x * cornerLength,
            y: c2.y + vC3c2.unit.y * cornerLength,
        };
        // Second control point
        let c3c2curveCP = {
            x: c2.x + vC3c2.unit.x * cpDistance,
            y: c2.y + vC3c2.unit.y * cpDistance,
        };

        // Limit floating point precision
        const limit = (point) => ({
            x: limitPrecision(point.x, 3),
            y: limitPrecision(point.y, 3),
        });

        c1c2curvePoint = limit(c1c2curvePoint);
        c1c2curveCP = limit(c1c2curveCP);
        c3c2curvePoint = limit(c3c2curvePoint);
        c3c2curveCP = limit(c3c2curveCP);

        // If at last coordinate of polygon, use the end of the curve as
        // the polygon starting point
        if (i === pathCoords.length - 1) {
            path.unshift(`M ${c3c2curvePoint.x} ${c3c2curvePoint.y}`);
        }

        // Draw line from previous point to the start of the curve
        path.push(`L ${c1c2curvePoint.x} ${c1c2curvePoint.y}`);

        // Cubic bezier to draw the actual curve
        path.push(
            `C ${c1c2curveCP.x} ${c1c2curveCP.y}, ${c3c2curveCP.x} ${c3c2curveCP.y}, ${c3c2curvePoint.x} ${c3c2curvePoint.y}`
        );
    }

    // Close path
    path.push("Z");

    return path.join(" ");
}

const line_interpolate = (widths) => {
    let left = [];
    let right = [];
    let nonzero_widths = widths.filter((d) => d > 0);
    let means = nonzero_widths.reduce((p, v) => p + v, 0) / nonzero_widths.length;
    if (widths[0] === 0) widths[0] = means;
    if (widths[widths.length - 1] === 0) widths[widths.length - 1] = means;
    for (let i = 0; i < widths.length; i++) {
        if (widths[i] > 0) left.push(i);
        else left.push(left[i - 1]);
    }
    right = left.map((d) => -1);
    for (let i = widths.length - 1; i >= 0; i--) {
        if (widths[i] > 0) right[i] = i;
        else right[i] = right[i + 1];
    }
    // right = right.reverse();
    for (let i = 0; i < widths.length; i++) {
        if (widths[i] === 0) {
            widths[i] =
                widths[left[i]] +
                ((i - left[i]) * (widths[right[i]] - widths[left[i]])) /
                (right[i] - left[i]);
        }
    }
    return widths;
};

// Properties of a line
// I:  - pointA (array) [x,y]: coordinates
//     - pointB (array) [x,y]: coordinates
// O:  - (object) { length: l, angle: a }: properties of the line
const line = (pointA, pointB) => {
    const lengthX = pointB[0] - pointA[0];
    const lengthY = pointB[1] - pointA[1];
    return {
        length: Math.sqrt(Math.pow(lengthX, 2) + Math.pow(lengthY, 2)),
        angle: Math.atan2(lengthY, lengthX),
    };
};

const smoothing = 0.2;
// Position of a control point
// I:  - current (array) [x, y]: current point coordinates
//     - previous (array) [x, y]: previous point coordinates
//     - next (array) [x, y]: next point coordinates
//     - reverse (boolean, optional): sets the direction
// O:  - (array) [x,y]: a tuple of coordinates
const controlPoint = (current, previous, next, reverse) => {
    // When 'current' is the first or last point of the array
    // 'previous' or 'next' don't exist.
    // Replace with 'current'
    const p = previous || current;
    const n = next || current;

    // Properties of the opposed-line
    const o = line(p, n);

    // If is end-control-point, add PI to the angle to go backward
    const angle = o.angle + (reverse ? Math.PI : 0);
    const length = o.length * smoothing;

    // The control point position is relative to the current point
    const x = current[0] + Math.cos(angle) * length;
    const y = current[1] + Math.sin(angle) * length;
    return [x, y];
};


// Create the bezier curve command
// I:  - point (array) [x,y]: current point coordinates
//     - i (integer): index of 'point' in the array 'a'
//     - a (array): complete array of points coordinates
// O:  - (string) 'C x2,y2 x1,y1 x,y': SVG cubic bezier C command
const bezierCommand = (point, i, a) => {
    // start control point
    const cps = controlPoint(a[i - 1], a[i - 2], point);
  
    // end control point
    const cpe = controlPoint(point, a[i - 1], a[i + 1], true);
    let res = `C ${cps[0]},${cps[1]} ${cpe[0]},${cpe[1]} ${point[0]},${point[1]}`;
    return res;
  };
  
  // const pairwise_line_generator = (x1, x2, y1, y2, y3, y4, r1 = 0.4, r2 = 0.6) => {
  const pairwise_line_generator = (point, i, a, r1 = 0.5, r2 = 0.5) => {
    let x1 = a[i - 1][0];
    let y1 = a[i - 1][1];
    let x2 = point[0];
    let y2 = point[1];
    let m1 = x1 + (x2 - x1) * r1;
    let m2 = x1 + (x2 - x1) * r2;
    // return `M${x1} ${y1}L${x1} ${y2} C${m1} ${y2} ${m2} ${y4} ${x2} ${y4} L${x2} ${y3} C${m2} ${y3} ${m1} ${y1} ${x1} ${y1} z`
    return `C${m1},${y1} ${m2},${y2} ${x2},${y2} `;
  };
  
  // const pairwise_line_generator = (x1, x2, y1, y2, y3, y4, r1 = 0.4, r2 = 0.6) => {
  const smooth_pairwise_line_generator = (point, i, a) => {
    let x1 = a[i - 1][0];
    let y1 = a[i - 1][1];
    let x2 = point[0];
    let y2 = point[1];
    let r1 = 0.4,
      r2 = 0.6;
    if (i > 1 && (y2 - y1) * (y1 - a[i - 2][1]) > 0) {
      r1 = 0.25;
      r2 = 0.75;
    }
    let m1 = x1 + (x2 - x1) * r1;
    let m2 = x1 + (x2 - x1) * r2;
    // return `M${x1} ${y1}L${x1} ${y2} C${m1} ${y2} ${m2} ${y4} ${x2} ${y4} L${x2} ${y3} C${m2} ${y3} ${m1} ${y1} ${x1} ${y1} z`
    return `C${m1},${y1} ${m2},${y2} ${x2},${y2} `;
  };

export {
    limitPrecision,
    statisticalDivision,
    scaleClustersBars, rotate, lowerIntersect, upperIntersect,
    areaMargin, placeImages, generateSortedEdges, vec,
    roundSvgPath, line_interpolate, line, controlPoint, bezierCommand,
    smooth_pairwise_line_generator, pairwise_line_generator
}