# @performance/dark-color-mode-check

Static repair references: 1. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_a9aeb4aa8bc19022`

Implementing dark mode can significantly reduce power consumption. Your application should adapt its UI based on the device's current theme settings.

### Triggering pattern

```arkts
src
├── main  
│   ├── ets    
│   └── resources
│       └── dark    
│           └── element
│           
├── mock
│   └── mock-config.json5
```

### Repair pattern

```arkts
src
├── main  
│   ├── ets    
│   └── resources
│       └── dark    
│           └── element
│               └── color.json     
│           
├── mock
│   └── mock-config.json5
```
