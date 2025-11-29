from typing import Iterator, Tuple, Optional
from ...domain.protocols import FrameProducer
from ...domain.entities import FrameAnalysis, Frame
from ..processors import FrameProcessor
from ....common.metrics import MetricsCollector

class VisionPipeline:
    """
    Orchestrates the computer vision pipeline:
    Source -> Detection -> Result
    """
    def __init__(
        self, 
        source: FrameProducer, 
        processor_chain: FrameProcessor,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.source = source
        self.processor_chain = processor_chain
        self.metrics_collector = metrics_collector

    def run(self) -> Iterator[Tuple[Frame, FrameAnalysis]]:
        """
        Runs the pipeline, yielding the current frame and the latest analysis.
        """
        analysis = None
        
        for frame in self.source:
            # Delegate processing to the chain
            analysis = self.processor_chain.process(frame, analysis)
            
            if self.metrics_collector:
                self.metrics_collector.increment_frames()
                
            yield frame, analysis

    def stop(self):
        """
        Stops the pipeline and releases resources.
        """
        self.source.release()
