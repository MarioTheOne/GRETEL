import re
class_parameters = "torch_geometric.nn.aggr.PowerMeanAggregation({'learn':True})"
class_only = "torch_geometric.nn.aggr.PowerMeanAggregation"

input = class_parameters
_cls_param_ptrn = re.compile('(^.*)'+ '\(' +'(.*)'+'\)')

res = _cls_param_ptrn.findall(input)

obj = input 
if len(res)==0:
    obj = input
else:
    obj =  [res[0][0],eval(res[0][1])]
 
print(obj)


