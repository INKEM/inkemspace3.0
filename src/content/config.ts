import { z, defineCollection } from 'astro:content'

const postsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.date(),
    lastMod: z.date().optional(),
    summary: z.string().optional(),
    cover: z.string().optional(),
    category: z.string().optional(),
    tags: z.array(z.string()).default([]),
    comments: z.boolean().default(true),
    draft: z.boolean().default(false),
    sticky: z.number().default(0),
  }),
})

const projectsCollection = defineCollection({
  type: 'data',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    image: z.string(),
    link: z.string().url(),
  }),
})

const specCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    comments: z.boolean().default(true),
  }),
})

const friendsCollection = defineCollection({
  type: 'data',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    avatar: z.string(),
    link: z.string().url(),
  }),
})

const galleryCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    album: z.string(),
    images: z.array(z.object({
      url: z.string().url(),
      alt: z.string().optional(),
      description: z.string().optional(),
    })).default([]),
    videos: z.array(z.object({
      url: z.string().url(),
      cover: z.string().url().optional(),
      title: z.string().optional(),
    })).default([]),
    date: z.date(),
    draft: z.boolean().default(false),
    comments: z.boolean().default(true),
  }),
})

const notesCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    tags: z.array(z.string()).default([]),
    date: z.date(),
    draft: z.boolean().default(false),
  }),
})

export const collections = {
  posts: postsCollection,
  projects: projectsCollection,
  spec: specCollection,
  friends: friendsCollection,
  gallery: galleryCollection,
  notes: notesCollection,
}
