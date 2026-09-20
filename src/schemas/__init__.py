"""Typed contracts shared across pipeline layers."""

from src.schemas.amazon import (
    AmazonEnrichedReviewRecord,
    AmazonProductRecord,
    AmazonReviewRecord,
)
from src.schemas.aspects import (
    AspectDiscoveryBatch,
    AspectDiscoveryBatchResult,
    AspectDiscoveryConfig,
    AspectMention,
    AspectNormalizationResult,
    AspectMerge,
    DiscoveryReview,
    ExcludedAspectCandidate,
    ReviewAspectObservation,
    load_aspect_discovery_config,
)
from src.schemas.dataset import (
    DatasetFile,
    DatasetManifest,
    DatasetSchemaVersions,
    DateWindow,
)
from src.schemas.workspace import (
    ProductNicheDefinition,
    SellerWorkspaceScope,
    load_product_niche_definition,
    load_seller_workspace_scope,
)

__all__ = [
    "AmazonEnrichedReviewRecord",
    "AmazonProductRecord",
    "AmazonReviewRecord",
    "AspectDiscoveryBatch",
    "AspectDiscoveryBatchResult",
    "AspectDiscoveryConfig",
    "AspectMention",
    "AspectNormalizationResult",
    "AspectMerge",
    "DiscoveryReview",
    "ExcludedAspectCandidate",
    "ReviewAspectObservation",
    "DatasetFile",
    "DatasetManifest",
    "DatasetSchemaVersions",
    "DateWindow",
    "ProductNicheDefinition",
    "SellerWorkspaceScope",
    "load_product_niche_definition",
    "load_seller_workspace_scope",
    "load_aspect_discovery_config",
]
