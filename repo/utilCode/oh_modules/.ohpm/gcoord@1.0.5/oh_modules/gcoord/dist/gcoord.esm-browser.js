/**
 * @preserve
 * gcoord 1.0.5, geographic coordinate library
 * Copyright (c) 2023 Jiulong Hu <me@hujiulong.com>
 */
const { sin: sin$1, cos: cos$1, sqrt: sqrt$1, abs: abs$1, PI: PI$1 } = Math;
const a = 6378245;
const ee = 0.006693421622965823;
// roughly check whether coordinates are in China.
function isInChinaBbox(lon, lat) {
    return lon >= 72.004 && lon <= 137.8347 && lat >= 0.8293 && lat <= 55.8271;
}
function transformLat(x, y) {
    let ret = -100 + 2 * x + 3 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * sqrt$1(abs$1(x));
    ret += ((20 * sin$1(6 * x * PI$1) + 20 * sin$1(2 * x * PI$1)) * 2) / 3;
    ret += ((20 * sin$1(y * PI$1) + 40 * sin$1((y / 3) * PI$1)) * 2) / 3;
    ret += ((160 * sin$1((y / 12) * PI$1) + 320 * sin$1((y * PI$1) / 30)) * 2) / 3;
    return ret;
}
function transformLon(x, y) {
    let ret = 300 + x + 2 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * sqrt$1(abs$1(x));
    ret += ((20 * sin$1(6 * x * PI$1) + 20 * sin$1(2 * x * PI$1)) * 2) / 3;
    ret += ((20 * sin$1(x * PI$1) + 40 * sin$1((x / 3) * PI$1)) * 2) / 3;
    ret += ((150 * sin$1((x / 12) * PI$1) + 300 * sin$1((x / 30) * PI$1)) * 2) / 3;
    return ret;
}
function delta(lon, lat) {
    let dLon = transformLon(lon - 105, lat - 35);
    let dLat = transformLat(lon - 105, lat - 35);
    const radLat = (lat / 180) * PI$1;
    let magic = sin$1(radLat);
    magic = 1 - ee * magic * magic;
    const sqrtMagic = sqrt$1(magic);
    dLon = (dLon * 180) / ((a / sqrtMagic) * cos$1(radLat) * PI$1);
    dLat = (dLat * 180) / (((a * (1 - ee)) / (magic * sqrtMagic)) * PI$1);
    return [dLon, dLat];
}
function WGS84ToGCJ02(coord) {
    const [lon, lat] = coord;
    if (!isInChinaBbox(lon, lat))
        return [lon, lat];
    const d = delta(lon, lat);
    return [lon + d[0], lat + d[1]];
}
function GCJ02ToWGS84(coord) {
    const [lon, lat] = coord;
    if (!isInChinaBbox(lon, lat))
        return [lon, lat];
    let [wgsLon, wgsLat] = [lon, lat];
    let tempPoint = WGS84ToGCJ02([wgsLon, wgsLat]);
    let dx = tempPoint[0] - lon;
    let dy = tempPoint[1] - lat;
    while (abs$1(dx) > 1e-6 || abs$1(dy) > 1e-6) {
        wgsLon -= dx;
        wgsLat -= dy;
        tempPoint = WGS84ToGCJ02([wgsLon, wgsLat]);
        dx = tempPoint[0] - lon;
        dy = tempPoint[1] - lat;
    }
    return [wgsLon, wgsLat];
}

const { sin, cos, atan2, sqrt, PI } = Math;
const baiduFactor = (PI * 3000.0) / 180.0;
function BD09ToGCJ02(coord) {
    const [lon, lat] = coord;
    const x = lon - 0.0065;
    const y = lat - 0.006;
    const z = sqrt(x * x + y * y) - 0.00002 * sin(y * baiduFactor);
    const theta = atan2(y, x) - 0.000003 * cos(x * baiduFactor);
    const newLon = z * cos(theta);
    const newLat = z * sin(theta);
    return [newLon, newLat];
}
function GCJ02ToBD09(coord) {
    const [lon, lat] = coord;
    const x = lon;
    const y = lat;
    const z = sqrt(x * x + y * y) + 0.00002 * sin(y * baiduFactor);
    const theta = atan2(y, x) + 0.000003 * cos(x * baiduFactor);
    const newLon = z * cos(theta) + 0.0065;
    const newLat = z * sin(theta) + 0.006;
    return [newLon, newLat];
}

// https://github.com/Turfjs/turf/blob/master/packages/turf-projection/index.ts
const R2D = 180 / Math.PI;
const D2R = Math.PI / 180;
const A = 6378137.0;
const MAXEXTENT = 20037508.342789244;
function ESPG3857ToWGS84(xy) {
    return [
        (xy[0] * R2D) / A,
        (Math.PI * 0.5 - 2.0 * Math.atan(Math.exp(-xy[1] / A))) * R2D,
    ];
}
function WGS84ToEPSG3857(lonLat) {
    // compensate longitudes passing the 180th meridian
    // from https://github.com/proj4js/proj4js/blob/master/lib/common/adjust_lon.js
    const adjusted = Math.abs(lonLat[0]) <= 180
        ? lonLat[0]
        : lonLat[0] - (lonLat[0] < 0 ? -1 : 1) * 360;
    const xy = [
        A * adjusted * D2R,
        A * Math.log(Math.tan(Math.PI * 0.25 + 0.5 * lonLat[1] * D2R)),
    ];
    // if xy value is beyond maxextent (e.g. poles), return maxextent
    if (xy[0] > MAXEXTENT)
        xy[0] = MAXEXTENT;
    if (xy[0] < -MAXEXTENT)
        xy[0] = -MAXEXTENT;
    if (xy[1] > MAXEXTENT)
        xy[1] = MAXEXTENT;
    if (xy[1] < -MAXEXTENT)
        xy[1] = -MAXEXTENT;
    return xy;
}

const { abs } = Math;
const MCBAND = [12890594.86, 8362377.87, 5591021, 3481989.83, 1678043.12, 0];
const LLBAND = [75, 60, 45, 30, 15, 0];
const MC2LL = [
    [
        1.410526172116255e-8, 0.00000898305509648872, -1.9939833816331,
        200.9824383106796, -187.2403703815547, 91.6087516669843, -23.38765649603339,
        2.57121317296198, -0.03801003308653, 17337981.2,
    ],
    [
        -7.435856389565537e-9, 0.000008983055097726239, -0.78625201886289,
        96.32687599759846, -1.85204757529826, -59.36935905485877, 47.40033549296737,
        -16.50741931063887, 2.28786674699375, 10260144.86,
    ],
    [
        -3.030883460898826e-8, 0.00000898305509983578, 0.30071316287616,
        59.74293618442277, 7.357984074871, -25.38371002664745, 13.45380521110908,
        -3.29883767235584, 0.32710905363475, 6856817.37,
    ],
    [
        -1.981981304930552e-8, 0.000008983055099779535, 0.03278182852591,
        40.31678527705744, 0.65659298677277, -4.44255534477492, 0.85341911805263,
        0.12923347998204, -0.04625736007561, 4482777.06,
    ],
    [
        3.09191371068437e-9, 0.000008983055096812155, 0.00006995724062,
        23.10934304144901, -0.00023663490511, -0.6321817810242, -0.00663494467273,
        0.03430082397953, -0.00466043876332, 2555164.4,
    ],
    [
        2.890871144776878e-9, 0.000008983055095805407, -3.068298e-8,
        7.47137025468032, -0.00000353937994, -0.02145144861037, -0.00001234426596,
        0.00010322952773, -0.00000323890364, 826088.5,
    ],
];
const LL2MC = [
    [
        -0.0015702102444, 111320.7020616939, 1704480524535203, -10338987376042340,
        26112667856603880, -35149669176653700, 26595700718403920,
        -10725012454188240, 1800819912950474, 82.5,
    ],
    [
        0.0008277824516172526, 111320.7020463578, 647795574.6671607,
        -4082003173.641316, 10774905663.51142, -15171875531.51559,
        12053065338.62167, -5124939663.577472, 913311935.9512032, 67.5,
    ],
    [
        0.00337398766765, 111320.7020202162, 4481351.045890365, -23393751.19931662,
        79682215.47186455, -115964993.2797253, 97236711.15602145,
        -43661946.33752821, 8477230.501135234, 52.5,
    ],
    [
        0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245,
        992013.7397791013, -1221952.21711287, 1340652.697009075, -620943.6990984312,
        144416.9293806241, 37.5,
    ],
    [
        -0.0003441963504368392, 111320.7020576856, 278.2353980772752,
        2485758.690035394, 6070.750963243378, 54821.18345352118, 9540.606633304236,
        -2710.55326746645, 1405.483844121726, 22.5,
    ],
    [
        -0.0003218135878613132, 111320.7020701615, 0.00369383431289,
        823725.6402795718, 0.46104986909093, 2351.343141331292, 1.58060784298199,
        8.77738589078284, 0.37238884252424, 7.45,
    ],
];
function transform$1(x, y, factors) {
    const cc = abs(y) / factors[9];
    let xt = factors[0] + factors[1] * abs(x);
    let yt = factors[2] +
        factors[3] * cc +
        factors[4] * Math.pow(cc, 2) +
        factors[5] * Math.pow(cc, 3) +
        factors[6] * Math.pow(cc, 4) +
        factors[7] * Math.pow(cc, 5) +
        factors[8] * Math.pow(cc, 6);
    xt *= x < 0 ? -1 : 1;
    yt *= y < 0 ? -1 : 1;
    return [xt, yt];
}
function BD09toBD09MC(coord) {
    const [lng, lat] = coord;
    let factors = [];
    for (let i = 0; i < LLBAND.length; i++) {
        if (abs(lat) > LLBAND[i]) {
            factors = LL2MC[i];
            break;
        }
    }
    return transform$1(lng, lat, factors);
}
function BD09MCtoBD09(coord) {
    const [x, y] = coord;
    let factors = [];
    for (let i = 0; i < MCBAND.length; i++) {
        if (y >= MCBAND[i]) {
            factors = MC2LL[i];
            break;
        }
    }
    return transform$1(x, y, factors);
}

function assert(condition, msg) {
    if (!condition)
        throw new Error(msg);
}
/**
 * isArray
 *
 * @param {*} input variable to validate
 * @returns {boolean} true/false
 */
function isArray(input) {
    return !!input && Object.prototype.toString.call(input) === '[object Array]';
}
/**
 * isNumber
 *
 * @param {*} num Number to validate
 * @returns {boolean} true/false
 * @example
 * isNumber(123)
 * //=true
 * isNumber('foo')
 * //=false
 */
function isNumber(input) {
    return !isNaN(Number(input)) && input !== null && !isArray(input);
}
/**
 * compose
 */
function compose(...funcs) {
    const start = funcs.length - 1;
    /* eslint-disable func-names */
    return function (...args) {
        let i = start;
        let result = funcs[start].apply(null, args);
        while (i--)
            result = funcs[i].call(null, result);
        return result;
    };
}
/**
 * Iterate over coordinates in any GeoJSON object, similar to Array.forEach()
 * https://github.com/Turfjs/turf/blob/master/packages/turf-meta/index.mjs
 *
 * @name coordEach
 * @param {FeatureCollection|Feature|Geometry} geojson any GeoJSON object
 * @param {Function} callback a method that takes (currentCoord, coordIndex, featureIndex, multiFeatureIndex)
 * @param {boolean} [excludeWrapCoord=false] whether or not to include the final coordinate of LinearRings that wraps the ring in its iteration.
 * @returns {void}
 * @example
 * let features = featureCollection([
 *   point([26, 37], {"foo": "bar"}),
 *   point([36, 53], {"hello": "world"})
 * ]);
 *
 * coordEach(features, function (currentCoord, coordIndex, featureIndex, multiFeatureIndex, geometryIndex) {
 *   //=currentCoord
 *   //=coordIndex
 *   //=featureIndex
 *   //=multiFeatureIndex
 *   //=geometryIndex
 * });
 */
/* eslint-disable no-param-reassign */
function coordEach(geojson, callback, excludeWrapCoord = false) {
    // Handles null Geometry -- Skips this GeoJSON
    if (geojson === null)
        return;
    /* eslint-disable-next-line */
    let j;
    let k;
    let l;
    let geometry;
    let coords;
    let stopG;
    let wrapShrink = 0;
    let coordIndex = 0;
    let geometryMaybeCollection;
    let isGeometryCollection;
    const { type } = geojson;
    const isFeatureCollection = type === 'FeatureCollection';
    const isFeature = type === 'Feature';
    const stop = isFeatureCollection
        ? geojson.features.length
        : 1;
    // This logic may look a little weird. The reason why it is that way
    // is because it's trying to be fast. GeoJSON supports multiple kinds
    // of objects at its root: FeatureCollection, Features, Geometries.
    // This function has the responsibility of handling all of them, and that
    // means that some of the `for` loops you see below actually just don't apply
    // to certain inputs. For instance, if you give this just a
    // Point geometry, then both loops are short-circuited and all we do
    // is gradually rename the input until it's called 'geometry'.
    //
    // This also aims to allocate as few resources as possible: just a
    // few numbers and booleans, rather than any temporary arrays as would
    // be required with the normalization approach.
    for (let featureIndex = 0; featureIndex < stop; featureIndex++) {
        geometryMaybeCollection = isFeatureCollection
            ? geojson.features[featureIndex].geometry
            : isFeature
                ? geojson.geometry
                : geojson;
        isGeometryCollection = geometryMaybeCollection
            ? geometryMaybeCollection.type === 'GeometryCollection'
            : false;
        stopG = isGeometryCollection
            ? geometryMaybeCollection.geometries.length
            : 1;
        for (let geomIndex = 0; geomIndex < stopG; geomIndex++) {
            let multiFeatureIndex = 0;
            let geometryIndex = 0;
            geometry = isGeometryCollection
                ? geometryMaybeCollection.geometries[geomIndex]
                : geometryMaybeCollection;
            // Handles null Geometry -- Skips this geometry
            if (geometry === null)
                continue;
            const geomType = geometry.type;
            wrapShrink =
                excludeWrapCoord &&
                    (geomType === 'Polygon' || geomType === 'MultiPolygon')
                    ? 1
                    : 0;
            switch (geomType) {
                case null:
                    break;
                case 'Point':
                    coords = geometry.coordinates;
                    if (callback(coords, coordIndex, featureIndex, multiFeatureIndex, geometryIndex) === false)
                        return false;
                    coordIndex++;
                    multiFeatureIndex++;
                    break;
                case 'LineString':
                case 'MultiPoint':
                    coords = geometry.coordinates;
                    for (j = 0; j < coords.length; j++) {
                        if (callback(coords[j], coordIndex, featureIndex, multiFeatureIndex, geometryIndex) === false)
                            return false;
                        coordIndex++;
                        if (geomType === 'MultiPoint')
                            multiFeatureIndex++;
                    }
                    if (geomType === 'LineString')
                        multiFeatureIndex++;
                    break;
                case 'Polygon':
                case 'MultiLineString':
                    coords = geometry.coordinates;
                    for (j = 0; j < coords.length; j++) {
                        for (k = 0; k < coords[j].length - wrapShrink; k++) {
                            if (callback(coords[j][k], coordIndex, featureIndex, multiFeatureIndex, geometryIndex) === false)
                                return false;
                            coordIndex++;
                        }
                        if (geomType === 'MultiLineString')
                            multiFeatureIndex++;
                        if (geomType === 'Polygon')
                            geometryIndex++;
                    }
                    if (geomType === 'Polygon')
                        multiFeatureIndex++;
                    break;
                case 'MultiPolygon':
                    coords = geometry.coordinates;
                    for (j = 0; j < coords.length; j++) {
                        geometryIndex = 0;
                        for (k = 0; k < coords[j].length; k++) {
                            for (l = 0; l < coords[j][k].length - wrapShrink; l++) {
                                if (callback(coords[j][k][l], coordIndex, featureIndex, multiFeatureIndex, geometryIndex) === false)
                                    return false;
                                coordIndex++;
                            }
                            geometryIndex++;
                        }
                        multiFeatureIndex++;
                    }
                    break;
                case 'GeometryCollection':
                    for (j = 0; j < geometry.geometries.length; j++) {
                        if (coordEach(geometry.geometries[j], callback, excludeWrapCoord) === false)
                            return false;
                    }
                    break;
                default:
                    throw new Error('Unknown Geometry Type');
            }
        }
    }
}

var CRSTypes;
(function (CRSTypes) {
    // WGS84
    CRSTypes["WGS84"] = "WGS84";
    CRSTypes["WGS1984"] = "WGS84";
    CRSTypes["EPSG4326"] = "WGS84";
    // GCJ02
    CRSTypes["GCJ02"] = "GCJ02";
    CRSTypes["AMap"] = "GCJ02";
    // BD09
    CRSTypes["BD09"] = "BD09";
    CRSTypes["BD09LL"] = "BD09";
    CRSTypes["Baidu"] = "BD09";
    CRSTypes["BMap"] = "BD09";
    // BD09MC
    CRSTypes["BD09MC"] = "BD09MC";
    CRSTypes["BD09Meter"] = "BD09MC";
    // EPSG3857
    CRSTypes["EPSG3857"] = "EPSG3857";
    CRSTypes["EPSG900913"] = "EPSG3857";
    CRSTypes["EPSG102100"] = "EPSG3857";
    CRSTypes["WebMercator"] = "EPSG3857";
    CRSTypes["WM"] = "EPSG3857";
})(CRSTypes || (CRSTypes = {}));
const WGS84 = {
    to: {
        [CRSTypes.GCJ02]: WGS84ToGCJ02,
        [CRSTypes.BD09]: compose(GCJ02ToBD09, WGS84ToGCJ02),
        [CRSTypes.BD09MC]: compose(BD09toBD09MC, GCJ02ToBD09, WGS84ToGCJ02),
        [CRSTypes.EPSG3857]: WGS84ToEPSG3857,
    },
};
const GCJ02 = {
    to: {
        [CRSTypes.WGS84]: GCJ02ToWGS84,
        [CRSTypes.BD09]: GCJ02ToBD09,
        [CRSTypes.BD09MC]: compose(BD09toBD09MC, GCJ02ToBD09),
        [CRSTypes.EPSG3857]: compose(WGS84ToEPSG3857, GCJ02ToWGS84),
    },
};
const BD09 = {
    to: {
        [CRSTypes.WGS84]: compose(GCJ02ToWGS84, BD09ToGCJ02),
        [CRSTypes.GCJ02]: BD09ToGCJ02,
        [CRSTypes.EPSG3857]: compose(WGS84ToEPSG3857, GCJ02ToWGS84, BD09ToGCJ02),
        [CRSTypes.BD09MC]: BD09toBD09MC,
    },
};
const EPSG3857 = {
    to: {
        [CRSTypes.WGS84]: ESPG3857ToWGS84,
        [CRSTypes.GCJ02]: compose(WGS84ToGCJ02, ESPG3857ToWGS84),
        [CRSTypes.BD09]: compose(GCJ02ToBD09, WGS84ToGCJ02, ESPG3857ToWGS84),
        [CRSTypes.BD09MC]: compose(BD09toBD09MC, GCJ02ToBD09, WGS84ToGCJ02, ESPG3857ToWGS84),
    },
};
const BD09MC = {
    to: {
        [CRSTypes.WGS84]: compose(GCJ02ToWGS84, BD09ToGCJ02, BD09MCtoBD09),
        [CRSTypes.GCJ02]: compose(BD09ToGCJ02, BD09MCtoBD09),
        [CRSTypes.EPSG3857]: compose(WGS84ToEPSG3857, GCJ02ToWGS84, BD09ToGCJ02, BD09MCtoBD09),
        [CRSTypes.BD09]: BD09MCtoBD09,
    },
};
const crsMap = {
    WGS84,
    GCJ02,
    BD09,
    EPSG3857,
    BD09MC,
};
var crsMap$1 = crsMap;

/**
 * transform
 *
 * @param {geojson|position|string} input
 * @returns {geojson|position} output
 */
/* eslint-disable no-param-reassign */
function transform(input, crsFrom, crsTo) {
    assert(!!input, 'The args[0] input coordinate is required');
    assert(!!crsFrom, 'The args[1] original coordinate system is required');
    assert(!!crsTo, 'The args[2] target coordinate system is required');
    if (crsFrom === crsTo)
        return input;
    const from = crsMap$1[crsFrom];
    assert(!!from, `Invalid original coordinate system: ${crsFrom}`);
    const to = from.to[crsTo];
    assert(!!to, `Invalid target coordinate system: ${crsTo}`);
    const type = typeof input;
    assert(type === 'string' || type === 'object', `Invalid input coordinate type: ${type}`);
    if (type === 'string') {
        try {
            input = JSON.parse(input);
        }
        catch (e) {
            throw new Error(`Invalid input coordinate: ${input}`);
        }
    }
    let isPosition = false;
    if (isArray(input)) {
        assert(input.length >= 2, `Invalid input coordinate: ${input}`);
        assert(isNumber(input[0]) && isNumber(input[1]), `Invalid input coordinate: ${input}`);
        input = input.map(Number);
        isPosition = true;
    }
    const convert = to;
    if (isPosition)
        return convert(input);
    // GeoJSON类型直接转换输入
    coordEach(input, (coord) => {
        [coord[0], coord[1]] = convert(coord);
    });
    return input;
}

const exported = Object.assign(Object.assign({}, CRSTypes), { // 兼容原来gcoord.WGS84的使用方式
    CRSTypes,
    transform });

export { exported as default };
