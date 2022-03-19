**definition of loss function**

1 use _def_ function

2 class

  (example: Dice Loss, use in segmentation)
  
  dice = 1-(2*|X∩Y|/(|X|+|Y|)). OR dice = (2 * tp) / (2 * tp + fp + fn)
  
  dice loss=1-dice
  
  To calculate similarity of two samples.
  
  ```
  class DiceLoss(nn.Module):
      def _init_():
          def forward():
          return
  criterion=DiceLoss()
  loss=criterion(input,targets)
  ```
**adjust learning rate**

  official: torch.optim.lr_scheduler
  
  Every time the optimizer is updated, use scheduler to adjust lr.
  
  define by yourself
  ```
  def function():
      lr=... # set by yourself
      for param_group in optimizer.param_groups:
      #optimizer.param_groups stores parameters in optimizer
          param_group['lr'] = lr #replace
  ```
**finetune**

  from transfer learning
  
  1 adjust parameters from pretrained, choose whether to keep weights(_pretrained = True/ False_)
  
  2 freeze some layers(_requires_grad = False_). Use nn.sequential get index, then change this layer.
  
**half precision**

  To get more batch size, change float32 to float16.
  
  three steps:
  from torch.cuda.amp import autocast
  def forward()
  with autocast(): #after input
  
