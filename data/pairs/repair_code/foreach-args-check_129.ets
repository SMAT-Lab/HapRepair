@Entry
@Component
struct App {
  build() {
    VStack() {
      ForEach(this.photos, (photo, index) => {
        PhotoThumbnail(photo);
      }, (photo) => photo.uniqueID);
    }
  }
}