import { SITE } from "@config";
import type { CollectionEntry } from "astro:content";

const postFilter = ({ data }: CollectionEntry<"blog">) => {
  const isPublishTimePassed = (SITE.scheduledPostMargin != null) ?
    (Date.now() >
    new Date(data.pubDatetime).getTime() - SITE.scheduledPostMargin) : true;
  return !data.draft && (import.meta.env.DEV || isPublishTimePassed);
};

export default postFilter;
