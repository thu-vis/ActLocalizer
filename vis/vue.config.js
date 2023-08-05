module.exports = {
  "transpileDependencies": [
    "vuetify"
  ],
  devServer: {
    port: 20222,
    proxy: {
      '/api': {
          target: 'http://localhost:20221/',
          changeOrigin: true,
          ws: true,
          pathRewrite: {
              '^/api': ''
          }
      }
    }
  },
  configureWebpack: {
    devtool: 'source-map'
  }
}