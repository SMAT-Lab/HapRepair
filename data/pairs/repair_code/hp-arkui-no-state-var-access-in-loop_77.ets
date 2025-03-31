let currentIndex = this.currentIndex
          for ( let indexNumber = 1 ; indexNumber < 101 ; indexNumber ++ ) {
            let message: string = apiItem.api!();
            apiItem.result = message;
            prompt.showToast({ message : message });
            currentIndex += 1;
            this.changeIndex = currentIndex;
            AppStorage.setOrCreate('stressNumber' , indexNumber);
            console.log(TAG , apiItem.method + "indexNumber is ------: " + indexNumber);
          }
          this.currentIndex = currentIndex