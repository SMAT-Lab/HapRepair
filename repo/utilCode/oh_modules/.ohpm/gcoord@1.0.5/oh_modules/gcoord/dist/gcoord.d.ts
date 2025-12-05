declare type BBox = BBox2d | BBox3d;

/**
 * Bounding box
 *
 * https://tools.ietf.org/html/rfc7946#section-5
 * A GeoJSON object MAY have a member named 'bbox' to include information on the coordinate range for its Geometries, Features, or FeatureCollections.
 * The value of the bbox member MUST be an array of length 2*n where n is the number of dimensions represented in the contained geometries,
 * with all axes of the most southwesterly point followed by all axes of the more northeasterly point.
 * The axes order of a bbox follows the axes order of geometries.
 */
declare type BBox2d = [number, number, number, number];

declare type BBox3d = [number, number, number, number, number, number];

export declare enum CRSTypes {
    WGS84 = "WGS84",
    WGS1984 = "WGS84",
    EPSG4326 = "WGS84",
    GCJ02 = "GCJ02",
    AMap = "GCJ02",
    BD09 = "BD09",
    BD09LL = "BD09",
    Baidu = "BD09",
    BMap = "BD09",
    BD09MC = "BD09MC",
    BD09Meter = "BD09MC",
    EPSG3857 = "EPSG3857",
    EPSG900913 = "EPSG3857",
    EPSG102100 = "EPSG3857",
    WebMercator = "EPSG3857",
    WM = "EPSG3857"
}

declare const exported: {
    CRSTypes: typeof CRSTypes;
    transform: typeof transform;
    WGS84: CRSTypes.WGS84;
    WGS1984: CRSTypes.WGS84;
    EPSG4326: CRSTypes.WGS84;
    GCJ02: CRSTypes.GCJ02;
    AMap: CRSTypes.GCJ02;
    BD09: CRSTypes.BD09;
    BD09LL: CRSTypes.BD09;
    Baidu: CRSTypes.BD09;
    BMap: CRSTypes.BD09;
    BD09MC: CRSTypes.BD09MC;
    BD09Meter: CRSTypes.BD09MC;
    EPSG3857: CRSTypes.EPSG3857;
    EPSG900913: CRSTypes.EPSG3857;
    EPSG102100: CRSTypes.EPSG3857;
    WebMercator: CRSTypes.EPSG3857;
    WM: CRSTypes.EPSG3857;
};
export default exported;

/**
 * Feature
 *
 * https://tools.ietf.org/html/rfc7946#section-3.2
 * A Feature object represents a spatially bounded thing.
 * Every Feature object is a GeoJSON object no matter where it occurs in a GeoJSON text.
 */
declare interface Feature<G = Geometry | GeometryCollection, P = Properties> extends GeoJSONObject {
    type: 'Feature';
    geometry: G;
    /**
     * A value that uniquely identifies this feature in a
     * https://tools.ietf.org/html/rfc7946#section-3.2.
     */
    id?: Id;
    /**
     * Properties associated with this feature.
     */
    properties: P;
}

/**
 * Feature Collection
 *
 * https://tools.ietf.org/html/rfc7946#section-3.3
 * A GeoJSON object with the type 'FeatureCollection' is a FeatureCollection object.
 * A FeatureCollection object has a member with the name 'features'.
 * The value of 'features' is a JSON array. Each element of the array is a Feature object as defined above.
 * It is possible for this array to be empty.
 */
declare interface FeatureCollection<G = Geometry | GeometryCollection, P = Properties> extends GeoJSONObject {
    type: 'FeatureCollection';
    features: Array<Feature<G, P>>;
}

/**
 * GeoJSON
 *
 * All GeoJSON objects
 */
export declare type GeoJSON = Feature | FeatureCollection | Geometry | GeometryCollection;

/**
 * GeoJSON Object
 *
 * https://tools.ietf.org/html/rfc7946#section-3
 * The GeoJSON specification also allows [foreign members](https://tools.ietf.org/html/rfc7946#section-6.1)
 * Developers should use '&' type in TypeScript or extend the interface to add these foreign members.
 */
declare interface GeoJSONObject {
    /**
     * Specifies the type of GeoJSON object.
     */
    type: string;
    /**
     * Bounding box of the coordinate range of the object's Geometries, Features, or Feature Collections.
     * https://tools.ietf.org/html/rfc7946#section-5
     */
    bbox?: BBox;
}

/**
 * Geometry
 *
 * https://tools.ietf.org/html/rfc7946#section-3
 */
declare interface Geometry extends GeoJSONObject {
    coordinates: Position | Position[] | Position[][] | Position[][][];
}

/**
 * GeometryCollection
 *
 * https://tools.ietf.org/html/rfc7946#section-3.1.8
 *
 * A GeoJSON object with type 'GeometryCollection' is a Geometry object.
 * A GeometryCollection has a member with the name 'geometries'.
 * The value of 'geometries' is an array.  Each element of this array is a GeoJSON Geometry object.
 * It is possible for this array to be empty.
 */
declare interface GeometryCollection extends GeometryObject {
    type: 'GeometryCollection';
    geometries: Array<Point | LineString | Polygon | MultiPoint | MultiLineString | MultiPolygon>;
}

/**
 * Geometry Object
 *
 * https://tools.ietf.org/html/rfc7946#section-3
 */
declare interface GeometryObject extends GeoJSONObject {
    type: GeometryTypes;
}

/**
 * GeometryTypes
 *
 * https://tools.ietf.org/html/rfc7946#section-1.4
 * The valid values for the 'type' property of GeoJSON geometry objects.
 */
declare type GeometryTypes = 'Point' | 'LineString' | 'Polygon' | 'MultiPoint' | 'MultiLineString' | 'MultiPolygon' | 'GeometryCollection';

/**
 * Id
 *
 * https://tools.ietf.org/html/rfc7946#section-3.2
 * If a Feature has a commonly used identifier, that identifier SHOULD be included as a member of
 * the Feature object with the name 'id', and the value of this member is either a JSON string or number.
 */
declare type Id = string | number;

/**
 * LineString Geometry Object
 *
 * https://tools.ietf.org/html/rfc7946#section-3.1.4
 */
declare interface LineString extends GeometryObject {
    type: 'LineString';
    coordinates: Position[];
}

/**
 * MultiLineString Geometry Object
 *
 * https://tools.ietf.org/html/rfc7946#section-3.1.5
 */
declare interface MultiLineString extends GeometryObject {
    type: 'MultiLineString';
    coordinates: Position[][];
}

/**
 * MultiPoint Geometry Object
 *
 * https://tools.ietf.org/html/rfc7946#section-3.1.3
 */
declare interface MultiPoint extends GeometryObject {
    type: 'MultiPoint';
    coordinates: Position[];
}

/**
 * MultiPolygon Geometry Object
 *
 * https://tools.ietf.org/html/rfc7946#section-3.1.7
 */
declare interface MultiPolygon extends GeometryObject {
    type: 'MultiPolygon';
    coordinates: Position[][][];
}

/**
 * Point Geometry Object
 *
 * https://tools.ietf.org/html/rfc7946#section-3.1.2
 */
declare interface Point extends GeometryObject {
    type: 'Point';
    coordinates: Position;
}

/**
 * Polygon Geometry Object
 *
 * https://tools.ietf.org/html/rfc7946#section-3.1.6
 */
declare interface Polygon extends GeometryObject {
    type: 'Polygon';
    coordinates: Position[][];
}

/**
 * Position
 *
 * https://tools.ietf.org/html/rfc7946#section-3.1.1
 * Array should contain between two and three elements.
 * The previous GeoJSON specification allowed more elements (e.g., which could be used to represent M values),
 * but the current specification only allows X, Y, and (optionally) Z to be defined.
 */
export declare type Position = [number, number] | [number, number, number];

/**
 * Properties
 *
 * https://tools.ietf.org/html/rfc7946#section-3.2
 * A Feature object has a member with the name 'properties'.
 * The value of the properties member is an object (any JSON object or a JSON null value).
 */
declare type Properties = {
    [name: string]: any;
} | null;

/**
 * transform
 *
 * @param {geojson|position|string} input
 * @returns {geojson|position} output
 */
declare function transform<T extends GeoJSON | Position>(input: T | string, crsFrom: CRSTypes, crsTo: CRSTypes): T;

export { }
