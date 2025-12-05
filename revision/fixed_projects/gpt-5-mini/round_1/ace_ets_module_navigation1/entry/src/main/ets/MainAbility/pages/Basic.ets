import { CommonNavigation } from './CommonNavigation';

export interface ClickItem {
  name: string;
  callback: () => void;
  id?: string;
}

@Entry
@Component
struct Basic {
  pathStack: NavPathStack = new NavPathStack()
  clickItem: Array<ClickItem> = [
    {
      name: 'push pageOne',
      callback: () => {
        this.pathStack.pushPath({name: 'pageOne'})
      },
      id: 'pushId'
    }
  ]

  build() {
    Column() {
      Scroll() {
        Column() {
          ForEach(this.clickItem, (item: ClickItem) => {
            Button(item.name).width(300).height(30).margin(10).onClick(item.callback).id(item.id)
          })
        }
      }.height('50%')
      CommonNavigation({pathStack: this.pathStack})
    }
  }
}