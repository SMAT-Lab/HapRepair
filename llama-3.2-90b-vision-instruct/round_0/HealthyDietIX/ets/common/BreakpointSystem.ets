import mediaQuery from '@ohos.mediaquery';

declare interface BreakPointTypeOption<T> {
  xs?: T
  sm?: T
  md?: T
  lg?: T
  xl?: T
  xxl?: T
}

export class BreakPointType<T> {
  options: BreakPointTypeOption<T>

  constructor(option: BreakPointTypeOption<T>) {
    this.options = option
  }

  getValue(currentBreakPoint: string) {
    if (currentBreakPoint === 'xs') {
      return this.options.xs
    } else if (currentBreakPoint === 'sm') {
      return this.options.sm
    } else if (currentBreakPoint === 'md') {
      return this.options.md
    } else if (currentBreakPoint === 'lg') {
      return this.options.lg
    } else if (currentBreakPoint === 'xl') {
      return this.options.xl
    } else if (currentBreakPoint === 'xxl') {
      return this.options.xxl
    } else {
      return undefined
    }
  }
}

type Breakpoint = {
  name: string
  condition: string
  mediaQueryListener?: mediaQuery.MediaQueryListener
}

export class BreakpointSystem {
  private currentBreakpoint: string = 'md'
  private breakpoints: Breakpoint[] = [
    { name: 'xs', condition: '(device-type: wearable)' },
    { name: 'sm', condition: '(device-type: default) and (orientation: portrait)' },
    { name: 'md', condition: '(device-type: default) and (orientation: landscape)' },
    { name: 'md', condition: '(device-type: tablet) and (orientation: portrait)' },
    { name: 'lg', condition: '(device-type: tablet) and (orientation: landscape)' }
  ]
  private updateCurrentBreakpoint(breakpoint: string) {
    if (this.currentBreakpoint !== breakpoint) {
      this.currentBreakpoint = breakpoint
      AppStorage.Set<string>('currentBreakpoint', this.currentBreakpoint)
    }
  }

  public register() {
    this.breakpoints.forEach((breakpoint: Breakpoint, index) => {
      let condition
      condition = breakpoint.condition
      breakpoint.mediaQueryListener = mediaQuery.matchMediaSync(condition)
      breakpoint.mediaQueryListener.on('change', (mediaQueryResult) => {
        if (mediaQueryResult.matches) {
          this.updateCurrentBreakpoint(breakpoint.name)
        }
      })
    })
  }

  public unregister() {
    this.breakpoints.forEach((breakpoint: Breakpoint) => {
      breakpoint.mediaQueryListener.off('change')
    })
  }
}