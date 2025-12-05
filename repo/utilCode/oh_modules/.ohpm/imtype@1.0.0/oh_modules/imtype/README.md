imagetype
=========

Get correct image type

## Support Image type
+ jpeg
+ png
+ gif
+ tiff
+ bmp 

- - -
## Usage
```
var imtype = require('imtype');
var fs = require('fs');

// isJPG;
var data = fs.readFileSync('foo.jpg');
console.log(imtype.isJPG(data));

// isPNG
var data = fs.readFileSync('foo.png');
console.log(imtype.isPNG(data));

// isGIF
var data = fs.readFileSync('foo.gif');
console.log(imtype.isGIF(data));

// isBMP
var data = fs.readFileSync('foo.bmp');
console.log(imtype.isBMP(data));

// isTIF
var data = fs.readFileSync('foo.tif');
console.log(imtype.isTIF(data));
```

## Test
```
make test
```