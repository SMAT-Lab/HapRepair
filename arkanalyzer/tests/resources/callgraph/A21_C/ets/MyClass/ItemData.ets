
export default class ItemData {
  title: Resource;
  img?: Resource;
  others?:Resource;

  constructor(title: Resource, img?: Resource,
              others?: Resource) {
    this.title = title;
    this.img = img;
    this.others = others;
  }
}