/**
 * Module imtype
 */

/**
 * Judge a image whether its type is bmp
 * @param  {Binary}  buf The image binary data
 * @return {Boolean}
 */
var isBMP = function (buf) {
  // BMP start of image is 0x424d

  if (!buf || buf.length < 2) {
    return false;
  }

  return buf[0] === 66 && buf[1] === 77;
};

/**
 * Judge a image whether its type is jpeg
 * @param  {Binary}  buf The image binary data
 * @return {Boolean}
 */
var isJPG = function (buf) {
  // JPEG start of image is 0xffd8

  if (!buf || buf.length < 2) {
    return false;
  }

  return buf[0] === 255 && buf[1] === 216;
};

/**
 * Judge a image whether its type is png
 * @param  {Binary}  buf The image binary data
 * @return {Boolean}
 */
var isPNG = function (buf) {
  // PNG start of image is 0x89504e470d0a1a0a

  if (!buf || buf.length < 8) {
    return false;
  }

  return buf[0] === 137 &&
         buf[1] === 80 &&
         buf[2] === 78 &&
         buf[3] === 71 &&
         buf[4] === 13 &&
         buf[5] === 10 &&
         buf[6] === 26 &&
         buf[7] === 10;
};

/**
 * Judge a image whether its type is gif
 * @param  {Binary}  buf The image binary data
 * @return {Boolean}
 */
var isGIF = function (buf) {
  // GIF start of image is 0x474946

  if (!buf || buf.length < 3) {
    return false;
  }

  return buf[0] === 71 &&
         buf[1] === 73 &&
         buf[2] === 70;
};

/**
 * Judge a image whether its type is tiff
 * @param  {Binary}  buf The image binary data
 * @return {Boolean}
 */
var isTIF = function (buf) {
  // TIFF start of image is 0x49492a00

  if (!buf || buf.length < 4) {
    return false;
  }

  return buf[0] === 73 &&
         buf[1] === 73 &&
         buf[2] === 42 &&
         buf[3] === 0;
};

module.exports = {
  isBMP: isBMP,
  isJPG: isJPG,
  isPNG: isPNG,
  isGIF: isGIF,
  isTIF: isTIF
};