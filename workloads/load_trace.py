from .alibaba_loader import AlibabaTrace

def get_trace_loader(trace_type, **kwargs):
    if trace_type.lower() == 'alibaba':
        csv = kwargs.get('csv')
        return AlibabaTrace(csv)
    # Dung trace khac thi elif them 
    return None
