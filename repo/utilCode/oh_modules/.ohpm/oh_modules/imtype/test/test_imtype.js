/**
 * Image unittest
 */

var assert = require('assert');
var fs = require('fs');
var imtype = require('../index');

describe('Test imtype', function () {
  describe('Test isBMP', function () {
    it('should be ok', function () {
      var data = fs.readFileSync(__dirname + '/1.bmp');
      var rst = imtype.isBMP(data);

      assert.equal(rst, true);
    });
  });

  describe('Test isPNG', function () {
    it('should be ok', function () {
      var data = fs.readFileSync(__dirname + '/1.png');
      var rst = imtype.isPNG(data);

      assert.equal(rst, true);
    });
  });

  describe('Test isJPG', function () {
    it('should be ok', function () {
      var data = fs.readFileSync(__dirname + '/1.jpg');
      var rst = imtype.isJPG(data);

      assert.equal(rst, true);
    });
  });

  describe('Test isGIF', function () {
    it('should be ok', function () {
      var data = fs.readFileSync(__dirname + '/1.gif');
      var rst = imtype.isGIF(data);

      assert.equal(rst, true);
    });
  });

  describe('Test isTIF', function () {
    it('should be ok', function () {
      var data = fs.readFileSync(__dirname + '/1.tif');
      var rst = imtype.isTIF(data);

      assert.equal(rst, true);
    });
  });
});