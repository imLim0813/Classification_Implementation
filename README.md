## Classification Implementation

### Description

<ul>
    <li> Dataset : CIFAR10</li>
    <ul>
        <li> Image size : Variable</li>
        <li> Mean : (0.485, 0.456, 0.406)</li>
        <li> Std : (0.229, 0.224, 0.225)</li>
        <li> Augmentation : Resize, Normalization, Horizontal Flip </li>
    </ul>
    <li> Specificiation</li>
    <ul>
        <li> Learning Rate : 1e-3 </li>
        <li> Loss function : Cross Entropy Loss </li>
        <li> Learning Rate Scheduler : CosineAnnealingWarmRestarts </li>
        
</ul>