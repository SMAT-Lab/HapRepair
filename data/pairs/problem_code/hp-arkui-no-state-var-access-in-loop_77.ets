for ( let indexNumber = 1 ; indexNumber < 101 ; indexNumber ++ ) {
            let message: string = apiItem.api!();
            apiItem.result = message;
            prompt.showToast({ message : message });
            this.currentIndex += 1;
            this.changeIndex = this.currentIndex;
            AppStorage.setOrCreate('stressNumber' , indexNumber);
            console.log(TAG , apiItem.method + "indexNumber is ------: " + indexNumber);
          }