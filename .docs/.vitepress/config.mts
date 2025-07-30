import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base:'/Movenet.Depth/',
  title: "Movenet with Depth integration",
  description: "Multi-modal learning in Human Pose Estimation",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Documentation', link: '/README.md' }
    ],

    sidebar: [
      {
        text: 'Examples',
        items: [
          { text: 'Overview of the models', link: '/README' },
          { text: 'Test the model', link: '/how-to-run' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/abdelha5/Movenet.Depth' }
    ]
  }
})
